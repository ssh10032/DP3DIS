import torch
import hydra
import math
import torch.nn as nn
import MinkowskiEngine.MinkowskiOps as me
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling
import numpy as np
from torch.nn import functional as F
from models.modules.common import conv
from models.position_embedding import PositionEmbeddingCoordsSine
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from models.modules.helpers_3detr import GenericMLP
from torch_scatter import scatter_mean, scatter_max, scatter_min
from torch.cuda.amp import autocast


class Mask3D(nn.Module):
    def __init__(self, config, hidden_dim, num_queries, num_heads, dim_feedforward,
                 sample_sizes, shared_decoder, num_classes,
                 num_decoders, dropout, pre_norm,
                 positional_encoding_type, non_parametric_queries, train_on_segments, normalize_pos_enc,
                 use_level_embed, scatter_type, hlevels,
                 use_np_features,
                 voxel_size,
                 max_sample_size,
                 random_queries,
                 gauss_scale,
                 random_query_both,
                 query_selection,
                 random_normal,
                 anchor_dim,
                 pos_embed_differ_each_layer=False
                 ):
        super().__init__()

        self.random_normal = random_normal
        self.random_query_both = random_query_both
        self.query_selection = query_selection
        self.random_queries = random_queries
        self.max_sample_size = max_sample_size
        self.gauss_scale = gauss_scale
        self.voxel_size = voxel_size
        self.scatter_type = scatter_type
        self.hlevels = hlevels
        self.use_level_embed = use_level_embed
        self.train_on_segments = train_on_segments
        self.normalize_pos_enc = normalize_pos_enc
        self.num_decoders = num_decoders
        self.num_classes = num_classes
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.shared_decoder = shared_decoder
        self.sample_sizes = sample_sizes
        self.non_parametric_queries = non_parametric_queries
        self.use_np_features = use_np_features
        self.mask_dim = hidden_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.pos_enc_type = positional_encoding_type
        self.pos_embed_differ_each_layer = False
        self.query_selection_flag = 0

        # 3D Anchor dimension = 6
        self.anchor_dim = anchor_dim

        self.backbone = hydra.utils.instantiate(config.backbone)
        self.num_levels = len(self.hlevels)

        # denoising parameter -songoh
        self.dn_num = 50
        self.noise_scale = 0.15
        self.label_enc = nn.Embedding(254, self.mask_dim)

        # denoise to bbox parameter
        self._bbox_embed = _bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        box_embed_layerlist = [_bbox_embed for i in range(self.num_decoders)]  # share box prediction each layer
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        # self.decoder.bbox_embed = self.bbox_embed

        sizes = self.backbone.PLANES[-5:]

        self.mask_features_head = conv(
            self.backbone.PLANES[7], self.mask_dim, kernel_size=1, stride=1, bias=True, D=3
        )

        if self.scatter_type == "mean":
            self.scatter_fn = scatter_mean
        elif self.scatter_type == "max":
            self.scatter_fn = lambda mask, p2s, dim: scatter_max(mask, p2s, dim=dim)[0]
        else:
            assert False, "Scatter function not known"

        assert (not use_np_features) or non_parametric_queries, "np features only with np queries"

        if self.non_parametric_queries:
            self.query_projection = GenericMLP(
                input_dim=self.mask_dim,
                hidden_dims=[self.mask_dim],
                output_dim=self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )

            if self.use_np_features:
                self.np_feature_projection = nn.Sequential(
                    nn.Linear(sizes[-1], hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
        elif self.random_query_both:            # False
            self.query_projection = GenericMLP(
                input_dim=2*self.mask_dim,
                hidden_dims=[2*self.mask_dim],
                output_dim=2*self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True
            )
        elif self.query_selection:
            self.ref_anchor_head = MLP(self.mask_dim, self.mask_dim, 3, 2)

            # projection anchor_dim to mask_dim -songoh
            self.query_projection = GenericMLP(
                input_dim=self.mask_dim,
                hidden_dims=[self.mask_dim],
                output_dim=self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )

            self.query_scale = MLP(self.mask_dim, self.mask_dim * 2, self.mask_dim * 3, 2)
            self.ca_qpos_sine_proj = nn.Linear(self.mask_dim * 3, self.mask_dim)
            self.pos_embed_differ_each_layer = pos_embed_differ_each_layer

            if pos_embed_differ_each_layer:
                # TO-DO : modify to have diffenrent weight each layer
                self.pos_refinement = nn.ModuleList(
                    [MLP(self.mask_dim, self.mask_dim, self.anchor_dim, 3) for i in range(6)])
                for pos_refinement in self.pos_refinement:
                    nn.init.constant_(pos_refinement.layers[-1].weight.data, 0)
                    nn.init.constant_(pos_refinement.layers[-1].bias.data, 0)
            else:
                self.pos_refinement = MLP(self.mask_dim, self.mask_dim, self.anchor_dim, 3)
                nn.init.constant_(self.pos_refinement.layers[-1].weight.data, 0)
                nn.init.constant_(self.pos_refinement.layers[-1].bias.data, 0)

            if self.use_np_features:
                self.np_feature_projection = nn.Sequential(
                    nn.Linear(sizes[-1], hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
        else:
            # PARAMETRIC QUERIES
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_pos = nn.Embedding(num_queries, hidden_dim)

        if self.use_level_embed:
            # learnable scale-level embedding
            self.level_embed = nn.Embedding(self.num_levels, hidden_dim)

        # instance query -> instance feature    -songoh
        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # instacne query -> semantic class  -songoh
        self.class_embed_head = nn.Linear(hidden_dim, self.num_classes)
        self.bbox_embed_head = MLP(hidden_dim, hidden_dim, 6, 3)

        # Query Selection Heads -Songoh
        self.enc_output = nn.Linear(self.mask_dim, hidden_dim)

        self.enc_output_norm = nn.LayerNorm(hidden_dim)

        if self.pos_enc_type == "legacy":
            self.pos_enc = PositionalEncoding3D(channels=self.mask_dim)
        elif self.pos_enc_type == "fourier":            # True
            self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier",
                                                       d_pos=self.mask_dim,
                                                       gauss_scale=self.gauss_scale,
                                                       normalize=self.normalize_pos_enc)
        elif self.pos_enc_type == "sine":
            self.pos_enc = PositionEmbeddingCoordsSine(pos_type="sine",
                                                       d_pos=self.mask_dim,
                                                       normalize=self.normalize_pos_enc)
        else:
            assert False, 'pos enc type not known'

        self.pooling = MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)

        self.masked_transformer_decoder = nn.ModuleList()
        self.cross_attention = nn.ModuleList()
        self.self_attention = nn.ModuleList()
        self.ffn_attention = nn.ModuleList()

        # #### position_encoding refinement -Songoh ####
        # self.pos_embed_differ_each_layer = pos_embed_differ_each_layer
        # if pos_embed_differ_each_layer:
        #     # TO-DO : modify to have diffenrent weight each layer
        #     self.pos_refinement = nn.ModuleList([MLP(self.mask_dim, self.mask_dim, 3, 3) for i in range(6)])
        #     for pos_refinement in self.pos_refinement:
        #         nn.init.constant_(pos_refinement.layers[-1].weight.data, 0)
        #         nn.init.constant_(pos_refinement.layers[-1].bias.data, 0)
        # else:
        #     self.pos_refinement = MLP(self.mask_dim, self.mask_dim, 3, 3)
        #     nn.init.constant_(self.pos_refinement.layers[-1].weight.data, 0)
        #     nn.init.constant_(self.pos_refinement.layers[-1].bias.data, 0)
        # #### position_encoding refinement -Songoh ####


        self.lin_squeeze = nn.ModuleList()

        num_shared = self.num_decoders if not self.shared_decoder else 1

        for _ in range(num_shared):
            tmp_cross_attention = nn.ModuleList()
            tmp_self_attention = nn.ModuleList()
            tmp_ffn_attention = nn.ModuleList()

            # position_encoding refinement
            tmp_pos_refinement = nn.ModuleList()

            tmp_squeeze_attention = nn.ModuleList()
            for i, hlevel in enumerate(self.hlevels):
                tmp_cross_attention.append(
                    CrossAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm
                    )
                )
                # Query refinement에서 cross attention할 때 128차원으로 Fr 차원 linear projection
                tmp_squeeze_attention.append(nn.Linear(sizes[hlevel], self.mask_dim))

                tmp_self_attention.append(
                    SelfAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm
                    )
                )

                tmp_ffn_attention.append(
                    FFNLayer(
                        d_model=self.mask_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm
                    )
                )


            self.cross_attention.append(tmp_cross_attention)
            self.self_attention.append(tmp_self_attention)
            self.ffn_attention.append(tmp_ffn_attention)
            self.lin_squeeze.append(tmp_squeeze_attention)

        self.decoder_norm = nn.LayerNorm(hidden_dim)

    def get_pos_encs(self, coords):
        pos_encodings_pcd = []

        for i in range(len(coords)):
            pos_encodings_pcd.append([[]])
            for coords_batch in coords[i].decomposed_features:
                scene_min = coords_batch.min(dim=0)[0][None, ...]
                scene_max = coords_batch.max(dim=0)[0][None, ...]

                with autocast(enabled=False):
                    tmp = self.pos_enc(coords_batch[None, ...].float(),
                                       input_range=[scene_min, scene_max])

                pos_encodings_pcd[-1][0].append(tmp.squeeze(0).permute((1, 0)))

        return pos_encodings_pcd

    def gen_sineembed_for_position(self, pos_tensor):
        # n_query, bs, _ = pos_tensor.size()
        # sineembed_tensor = torch.zeros(n_query, bs, 256)
        scale = 2 * math.pi
        dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / 128)
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        z_embed = pos_tensor[:, :, 2] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_z = z_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3).flatten(2)
        # 3D translation Positional Encoding
        if pos_tensor.size(-1) == 3:
            pos = torch.cat((pos_x, pos_y, pos_z), dim=2)
            # print('check1')
        # Anchor Box Positional Encoding
        elif pos_tensor.size(-1) == 6:
            w_embed = pos_tensor[:, :, 3] * scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

            h_embed = pos_tensor[:, :, 4] * scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

            d_embed = pos_tensor[:, :, 5] * scale
            pos_d = d_embed[:, :, None] / dim_t
            pos_d = torch.stack((pos_d[:, :, 0::2].sin(), pos_d[:, :, 1::2].cos()), dim=3).flatten(2)

            pos = torch.cat((pos_x, pos_y, pos_z, pos_w, pos_h, pos_d), dim=2)
            # print('check2')
        else:
            raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
        return pos

    def prepare_for_dn(self, targets, tgt, refpoint_emb, batch_size):
        """
        modified from dn-detr. You can refer to dn-detr
        https://github.com/IDEA-Research/DN-DETR/blob/main/models/dn_dab_deformable_detr/dn_components.py
        for more details
            :param dn_args: scalar, noise_scale
            :param tgt: original tgt (content) in the matching part
            :param refpoint_emb: positional anchor queries in the matching part
            :param batch_size: bs
            """
        if self.training:
            scalar, noise_scale = self.dn_num, self.noise_scale

            known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
            know_idx = [torch.nonzero(t) for t in known]
            known_num = [sum(k) for k in known]
            single_pad = int(max(known_num))
            # print('check')
            # use fix number of dn queries
            if max(known_num) > 0:
                # dn_num에서 batch마다 들어온 데이터 수 중 젤 큰수로 나눠줌
                scalar = scalar // (int(max(known_num)))
            else:
                scalar = 0
            if scalar == 0:
                # print('check')
                input_query_label = None
                input_query_bbox = None
                attn_mask = None
                mask_dict = None
                return input_query_label, input_query_bbox, attn_mask, mask_dict

            # can be modified to selectively denosie some label or boxes; also known label prediction
            unmask_bbox = unmask_label = torch.cat(known)
            labels = torch.cat([t['labels'] for t in targets])
            boxes = torch.cat([t['boxes'] for t in targets])
            batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
            # known
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)

            # noise
            # batch 3으로 들어온 데이터들 다 합치고 scalar 만큼 곱해줌
            known_indice = known_indice.repeat(scalar, 1).view(-1)
            known_labels = labels.repeat(scalar, 1).view(-1)
            known_bid = batch_idx.repeat(scalar, 1).view(-1)
            known_bboxs = boxes.repeat(scalar, 1)
            known_labels_expaned = known_labels.clone()
            known_bbox_expand = known_bboxs.clone()

            # noise on the label
            if noise_scale > 0:
                p = torch.rand_like(known_labels_expaned.float())
                chosen_indice = torch.nonzero(p < (noise_scale * 0.5)).view(-1)  # half of bbox prob
                new_label = torch.randint_like(chosen_indice, 0, self.num_classes)  # randomly put a new one here
                known_labels_expaned.scatter_(0, chosen_indice, new_label)
            # noise on the BBox/Anchor of Mask
            if noise_scale > 0:
                diff = torch.zeros_like(known_bbox_expand)
                diff[:, :3] = known_bbox_expand[:, 3:] / 2
                diff[:, 3:] = known_bbox_expand[:, 3:]
                known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
                                               diff).cuda() * noise_scale
                known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

            m = known_labels_expaned.long().to('cuda')
            input_label_embed = self.label_enc(m)
            input_bbox_embed = inverse_sigmoid(known_bbox_expand)

            pad_size = single_pad * scalar

            padding_label = torch.zeros(pad_size, self.mask_dim).cuda()
            padding_bbox = torch.zeros(pad_size, 6).cuda()

            if not refpoint_emb is None:
                input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
                input_query_bbox = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
            else:
                input_query_label = padding_label.repeat(batch_size, 1, 1)
                input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

            # map
            map_known_indice = torch.tensor([]).to('cuda')
            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
            if len(known_bid):
                input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
                input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

            tgt_size = pad_size + self.num_queries
            attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size,
                'scalar': scalar,
            }
        # else:
        #     if not refpoint_emb is None:
        #         input_query_label = tgt.repeat(batch_size, 1, 1)
        #         input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
        #     else:
        #         input_query_label = None
        #         input_query_bbox = None
        #     attn_mask = None
        #     mask_dict = None

        # 100*batch*256
        if not input_query_bbox is None:
            input_query_label = input_query_label
            input_query_bbox = input_query_bbox

        if input_query_bbox is None:
            print('nontype bbox!!!!!')

        # input_query_bbox = input_query_bbox.sigmoid().permute(1, 0, 2)
        # input_query_bbox = self.gen_sineembed_for_position(input_query_bbox).permute((1, 2, 0))
        # input_query_bbox = self.query_projection(input_query_bbox)
        # # query_sine_embed = query_sine_embed.permute((2, 0, 1))
        # input_query_bbox = input_query_bbox.permute((2, 0, 1))

        return input_query_label, input_query_bbox, attn_mask, mask_dict

    # def dn_post_process(self,outputs_class, outputs_coord, mask_dict,outputs_mask):
    #     """
    #         post process of dn after output from the transformer
    #         put the dn part in the mask_dict
    #         """
    #     assert mask_dict['pad_size'] > 0
    #     output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
    #     outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
    #     output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
    #     outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
    #     if outputs_mask is not None:
    #         output_known_mask = outputs_mask[:, :, :mask_dict['pad_size'], :]
    #         outputs_mask = outputs_mask[:, :, mask_dict['pad_size']:, :]
    #     out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1],'pred_masks': output_known_mask[-1]}
    #
    #     out['aux_outputs'] = self._set_aux_loss(output_known_class, output_known_mask,output_known_coord)
    #     mask_dict['output_known_lbs_bboxes']=out
    #     return outputs_class, outputs_coord, outputs_mask

    def dn_post_process(self,outputs_class, mask_dict,outputs_mask):
        """
            post process of dn after output from the transformer
            put the dn part in the mask_dict
            """
        assert mask_dict['pad_size'] > 0
        output_known_class = outputs_class[:, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, mask_dict['pad_size']:, :]

        output_known_mask_list = []
        outputs_mask_list = []
        if outputs_mask is not None:
            # output_known_mask = outputs_mask[:, :, :mask_dict['pad_size'], :]
            # outputs_mask = outputs_mask[:, :, mask_dict['pad_size']:, :]
            for i in range(len(outputs_mask)):
                output_known_mask = outputs_mask[i][:, :mask_dict['pad_size']]
                outputs_masks = outputs_mask[i][:, mask_dict['pad_size']:]
                output_known_mask_list.append(output_known_mask)
                outputs_mask_list.append(outputs_masks)

        out = {'pred_logits': output_known_class,'pred_masks': output_known_mask_list}

        out['aux_outputs'] = self._set_aux_loss(output_known_class, output_known_mask_list)
        mask_dict['output_known_lbs_bboxes']=out
        return outputs_class,  outputs_mask_list


    def make_dn_query(self, query, query_pos, mask_dict):
        assert mask_dict['pad_size'] > 0
        dn_query = query[:, :mask_dict['pad_size'], :]
        dn_query_pos = query_pos[:mask_dict['pad_size'],:, :]

        return dn_query, dn_query_pos

    def pred_box(self, query_pose, quries, ref0=None):
        """
        :param reference: reference box coordinates from each decoder layer
        :param hs: content
        :param ref0: whether there are prediction from the first layer
        """
        device = query_pose[0].device

        if ref0 is None:
            outputs_coord_list = []
        else:
            outputs_coord_list = [ref0.to(device)]
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(query_pose.permute((1, 0, 2)), self.bbox_embed, quries)):
            layer_delta_unsig = layer_bbox_embed(layer_hs).to(device)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig).to(device)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)
        return outputs_coord_list

    def forward(self, x, dn_data=None, point2segment=None, raw_coordinates=None, is_eval=False):
        # dn_data DN 쿼리 샘플링에 필요한 GT Label, BBox, Mask 데이터들 list
        #PCD : Point Cloud Data (164833, 96)
        #aux : auxiliary. list: 5

        # 추출된 포인트 특징들
        pcd_features, aux = self.backbone(x)

        batch_size = len(x.decomposed_coordinates)
        # if self.training:
        #     prediction_bbox = []
        predictions_class = []
        predictions_mask = []

        with torch.no_grad():
            # 3차원 포인트 좌표값들
            coordinates = me.SparseTensor(features=raw_coordinates,
                                          coordinate_manager=aux[-1].coordinate_manager,
                                          coordinate_map_key=aux[-1].coordinate_map_key,
                                          device=aux[-1].device)

            coords = [coordinates]
            for _ in reversed(range(len(aux)-1)):
                coords.append(self.pooling(coords[-1]))

            coords.reverse()

        # aux -> coords -> position encoding
        # list : 5
        pos_encodings_pcd = self.get_pos_encs(coords)

        # pcd_feature -> mask feature (N, 128)
        # mask feature is F0 in paper
        mask_features = self.mask_features_head(pcd_features)

        if self.train_on_segments:  # False
            mask_segments = []
            for i, mask_feature in enumerate(mask_features.decomposed_features):
                mask_segments.append(self.scatter_fn(mask_feature, point2segment[i], dim=0))

        sampled_coords = None

        # 쿼리 초기화
        # 1. Zeros + FPS
        # 2. Zeros + Random
        # 3. Random + Random
        # 4. Query Selection(Top K Encoder Feature + Anchor from Encoder Feature)
        if self.non_parametric_queries:     # True
            fps_idx = [furthest_point_sample(x.decomposed_coordinates[i][None, ...].float(),
                                             self.num_queries).squeeze(0).long()
                       for i in range(len(x.decomposed_coordinates))]

            sampled_coords = torch.stack([coordinates.decomposed_features[i][fps_idx[i].long(), :]
                                          for i in range(len(fps_idx))])
            # sample0 = sampled_coords

            mins = torch.stack([coordinates.decomposed_features[i].min(dim=0)[0] for i in range(len(coordinates.decomposed_features))])
            maxs = torch.stack([coordinates.decomposed_features[i].max(dim=0)[0] for i in range(len(coordinates.decomposed_features))])

            query_pos = self.pos_enc(sampled_coords.float(),
                                     input_range=[mins, maxs]
                                     )  # Batch, Dim, queries
            query_pos = self.query_projection(query_pos)

            if not self.use_np_features:            # True
                queries = torch.zeros_like(query_pos).permute((0, 2, 1))
            else:
                queries = torch.stack([pcd_features.decomposed_features[i][fps_idx[i].long(), :]
                                       for i in range(len(fps_idx))])
                queries = self.np_feature_projection(queries)
            # query_pos_songoh = query_pos
            query_pos = query_pos.permute((2, 0, 1))
        elif self.random_queries:       # False
            query_pos = torch.rand(batch_size, self.mask_dim, self.num_queries, device=x.device) - 0.5

            queries = torch.zeros_like(query_pos).permute((0, 2, 1))
            query_pos = query_pos.permute((2, 0, 1))
        elif self.random_query_both:        # False
            if not self.random_normal:
                query_pos_feat = torch.rand(batch_size, 2*self.mask_dim, self.num_queries, device=x.device) - 0.5
            else:
                query_pos_feat = torch.randn(batch_size, 2 * self.mask_dim, self.num_queries, device=x.device)

            queries = query_pos_feat[:, :self.mask_dim, :].permute((0, 2, 1))
            query_pos = query_pos_feat[:, self.mask_dim:, :].permute((2, 0, 1))
        elif self.query_selection:

            mins = torch.stack(
                [coordinates.decomposed_features[i].min(dim=0)[0] for i in range(len(coordinates.decomposed_features))])
            maxs = torch.stack(
                [coordinates.decomposed_features[i].max(dim=0)[0] for i in range(len(coordinates.decomposed_features))])
            self.query_selection_flag = 1
            enc_outputs_class_unselected = []
            enc_outputs_coord_unselected = []

            test = aux[2].decomposed_features
            for i in range(len(test)):
                classoutput = self.class_embed_head(test[i])
                coordoutput = self.bbox_embed_head(test[i])
                enc_outputs_class_unselected.append(classoutput)
                enc_outputs_coord_unselected.append(coordoutput)

            topk = self.num_queries
            tgt_group = []
            proposal_group = []
            for i in range(len(test)):
                topk_proposals = torch.topk(enc_outputs_class_unselected[i].max(-1)[0], topk, dim=0)[1]

                refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected[i], 0,
                                                       topk_proposals.unsqueeze(-1).repeat(1, 6))  # unsigmoid

                # Anchors from Top K selected  Encoder Feature -Songoh
                # refpoint_embed = refpoint_embed_undetach.detach()
                # Top K Selected Encoder Feature -Songoh
                # Top K index 뽑은 거 Mask Feature Dimension(self.mask_dim) 만큼 repeat시키고 해당 Feature 값 그대로 가져옴
                tgt_undetach = torch.gather(test[i], 0,
                                            topk_proposals.unsqueeze(-1).repeat(1, self.mask_dim)).unsqueeze(0)
                tgt_group.append(tgt_undetach)
                proposal_group.append(refpoint_embed_undetach)
            tgt_undetachs = torch.cat(tgt_group, dim=0)




            # Top K selected Encoder Feature Prediction -Songoh
            if self.train_on_segments:
                output_class, outputs_mask, attn_mask = self.mask_module(tgt_undetachs,
                                                              mask_features,
                                                              mask_segments,
                                                              0,
                                                              ret_attn_mask=True,
                                                              point2segment=point2segment,
                                                              coords=coords)
            else:
                output_class, outputs_mask, attn_mask = self.mask_module(tgt_undetachs,
                                                              mask_features,
                                                              None,
                                                              0,
                                                              ret_attn_mask=True,
                                                              point2segment=None,
                                                              coords=coords)


            self.query_selection_flag = 0

            predictions_class.append(output_class)
            predictions_mask.append(outputs_mask)

            decomposed_mask_coord = []
            decomposed_mask_coord = coords[4].decomposed_features
            # attn_mask = attn_mask.permute(1, 0)
            # index = torch.where(attn_mask==False)

            # list of mask tensor(True/False)
            decomposed_attn_mask_feature = attn_mask.decomposed_features
            # print('fuck u')

            # 0, 1, 2
            true_true_indice = []
            for i in range(len(decomposed_attn_mask_feature)):
                true_indice = []
                # permute mask to (150, K)
                decomposed_attn_mask_feature[i] = decomposed_attn_mask_feature[i].permute(1, 0)
                # if torch.all(decomposed_attn_mask_feature[i]) or torch.all(~decomposed_attn_mask_feature[i]):
                #     # print('debug msg')
                #     # print('debug msg 2')
                #     continue
                # else:
                #     for j in range(decomposed_attn_mask_feature[i].shape[0]):
                #         indice = (decomposed_attn_mask_feature[i][j]==False).nonzero(as_tuple=True)
                #         true_indice.append(indice[0].tolist())
                # # true_true_indice.append(true_indice)
                #     true_true_indice.append(true_indice)
                for j in range(decomposed_attn_mask_feature[i].shape[0]):
                    # sigmoid() > 0.5인 index만 추출 (0.5보다 큰 경우가 False)
                    indice = (decomposed_attn_mask_feature[i][j]==False).nonzero(as_tuple=True)
                    true_indice.append(indice[0].tolist())
                # true_true_indice.append(true_indice)
                true_true_indice.append(true_indice)

            masks_coord_list = []
            for i in range(len(true_true_indice)):
                mask_coord_list = []
                # 0 ~ 149
                for j in range(len(true_true_indice[i])):
                    mask_coord_list.append(decomposed_mask_coord[i][true_true_indice[i][j]])
                masks_coord_list.append(mask_coord_list)

            masks_xyzwhd_list = []
            for i in range(len(masks_coord_list)):
                mask_xyzwhd_list = []
                for j in range(len(masks_coord_list[i])):
                    # 각 차원에서 최소값과 최대값
                    if masks_coord_list[i][j].shape == (0,3):
                        ## ccccc:
                        mask_xyzwhd_list.append(proposal_group[i][j][:3])
                    else:
                        min_values, _ = torch.min(masks_coord_list[i][j], dim=0)
                        max_values, _ = torch.max(masks_coord_list[i][j], dim=0)

                        # 가로, 세로, 높이 계산
                        #### 6D Anchor PE ####
                        # width = max_values[0] - min_values[0]
                        # height = max_values[1] - min_values[1]
                        # depth = max_values[2] - min_values[2]
                        #### 6D Anchor PE ####

                        # 중심점 계산
                        center = (max_values + min_values) / 2
                        #### 6D Anchor PE ####
                        # mask_info = [center[0], center[1], center[2], width, height, depth]
                        #### 6D Anchor PE ####

                        #### 3D Coord PE ####
                        mask_info = [center[0], center[1], center[2]]
                        #### 3D Coord PE ####

                        mask_xyzwhd = torch.tensor(mask_info).to(x.device)
                        # mask_xyzwhd_list.append([x_mid, y_mid, z_mid, width, height, depth])
                        # mask_wyzwhd = torch.cat(center, width, height, depth, dim=0)
                        mask_xyzwhd_list.append(mask_xyzwhd)
                    # print('check')
                # print(mask_xyzwhd_list)
                mask_xyzwhd_list = torch.stack(mask_xyzwhd_list)
                masks_xyzwhd_list.append(mask_xyzwhd_list)


            obj_anchor = torch.stack(masks_xyzwhd_list, dim=0)
            # obj_anchor = mask_proposal.sigmoid().permute(1,0,2)
            # obj_anchor = mask_proposal
            query_sine_embed = self.pos_enc(obj_anchor, input_range=[mins, maxs])
            query_pos = self.query_projection(query_sine_embed)
            # query_sine_embed = query_sine_embed.permute((2, 0, 1))
            query_pos = query_pos.permute((2, 0, 1))
            queries = tgt_undetachs

            # print('check')
            # 딕셔너리로 인덱스 뽑는거는 이중 포문으로 연산량 너무 많이 듬 다른 방식으로 구현하기 8/1 songoh
            # if torch.all(decomposed_attn_mask_feature) or torch.all(~decomposed_attn_mask_feature):
            #     pass
            # else:
            #     list_dict = {}
            #     for i in range(len(decomposed_attn_mask_feature)):
            #         list_dict[f"idx_mask{i}"] = []
            #         for j in range(len(decomposed_attn_mask_feature[i])):
            #             list_dict[f"idx_mask{i}"].append(torch.nonzero(decomposed_attn_mask_feature[i][j]))
            #     print('check')
            # print('check')




        else:               # False
            # PARAMETRIC QUERIES
            queries = self.query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            query_pos = self.query_pos.weight.unsqueeze(1).repeat(1, batch_size, 1)


        # if self.dn != "no" and self.training:
        # DN Query 샘플링, 학습 시에만 적용됨
        if self.training:
            input_query_label, input_query_bbox, tgt_mask, mask_dict = \
                self.prepare_for_dn(dn_data, None, None, batch_size)

            if input_query_label is not None and input_query_bbox is not None:
                input_query_bbox = input_query_bbox[:,:,:3]
                # input_query_bbox = input_query_bbox.sigmoid().permute(1, 0, 2)
                obj_anchor = torch.cat([input_query_bbox, obj_anchor], dim=1)

                ##### 6D Anchor #####
                # input_query_bbox = self.gen_sineembed_for_position(input_query_bbox).permute((1, 2, 0))
                ##### 6D Anchor #####

                ##### 3D Coord #####
                # input_query_bbox = self.pos_enc(input_query_bbox, input_range=[min, max]).permute((1, 2, 0))
                input_query_bbox = self.pos_enc(input_query_bbox, input_range=[mins, maxs])
                input_query_bbox = self.query_projection(input_query_bbox)
                ##### 3D Coord #####

                # query_sine_embed = query_sine_embed.permute((2, 0, 1))
                input_query_bbox = input_query_bbox.permute((2, 0, 1))
                if mask_dict is not None:
                    queries = torch.cat([input_query_label, queries],dim=1)
                    query_pos = torch.cat([input_query_bbox, query_pos], dim=0)



        for decoder_counter in range(self.num_decoders):
            if self.shared_decoder:
                decoder_counter = 0
            # hlevels: 4 -> maybe feature resolution level
            for i, hlevel in enumerate(self.hlevels):
                if self.train_on_segments:
                    # cross attention에 사용할 attn mask 산출
                    output_class, outputs_mask, attn_mask = self.mask_module(queries,
                                                          mask_features,
                                                          mask_segments,
                                                          len(aux) - hlevel - 1,
                                                          ret_attn_mask=True,
                                                          point2segment=point2segment,
                                                          coords=coords)
                else:
                    # cross attention에 사용할 attn mask 산출
                    output_class, outputs_mask, attn_mask = self.mask_module(queries,
                                                          mask_features,
                                                          None,
                                                          len(aux) - hlevel - 1,
                                                          ret_attn_mask=True,
                                                          point2segment=None,
                                                          coords=coords)

                decomposed_aux = aux[hlevel].decomposed_features
                decomposed_attn = attn_mask.decomposed_features

                curr_sample_size = max([pcd.shape[0] for pcd in decomposed_aux])

                if min([pcd.shape[0] for pcd in decomposed_aux]) == 1:
                    raise RuntimeError("only a single point gives nans in cross-attention")

                if not (self.max_sample_size or is_eval):
                    curr_sample_size = min(curr_sample_size, self.sample_sizes[hlevel])

                rand_idx = []
                mask_idx = []
                for k in range(len(decomposed_aux)):
                    pcd_size = decomposed_aux[k].shape[0]
                    if pcd_size <= curr_sample_size:
                        # we do not need to sample
                        # take all points and pad the rest with zeroes and mask it
                        idx = torch.zeros(curr_sample_size,
                                          dtype=torch.long,
                                          device=queries.device)

                        midx = torch.ones(curr_sample_size,
                                          dtype=torch.bool,
                                          device=queries.device)

                        idx[:pcd_size] = torch.arange(pcd_size,
                                                      device=queries.device)

                        midx[:pcd_size] = False  # attend to first points
                    else:
                        # we have more points in pcd as we like to sample
                        # take a subset (no padding or masking needed)
                        idx = torch.randperm(decomposed_aux[k].shape[0],
                                             device=queries.device)[:curr_sample_size]
                        midx = torch.zeros(curr_sample_size,
                                           dtype=torch.bool,
                                           device=queries.device)  # attend to all

                    rand_idx.append(idx)
                    mask_idx.append(midx)

                batched_aux = torch.stack([
                    decomposed_aux[k][rand_idx[k], :] for k in range(len(rand_idx))
                ])

                batched_attn = torch.stack([
                    decomposed_attn[k][rand_idx[k], :] for k in range(len(rand_idx))
                ])

                batched_pos_enc = torch.stack([
                    pos_encodings_pcd[hlevel][0][k][rand_idx[k], :] for k in range(len(rand_idx))
                ])

                batched_attn.permute((0, 2, 1))[batched_attn.sum(1) == rand_idx[0].shape[0]] = False

                m = torch.stack(mask_idx)
                batched_attn = torch.logical_or(batched_attn, m[..., None])     # (1, 947, 100)
                # n-th decoder + hlevel???
                src_pcd = self.lin_squeeze[decoder_counter][i](batched_aux.permute((1, 0, 2)))
                if self.use_level_embed:
                    src_pcd += self.level_embed.weight[i]

                # 포인트 특징과 cross attention
                # cross attention in Query Refinement
                output = self.cross_attention[decoder_counter][i](
                    queries.permute((1, 0, 2)),             # (1, 100, 128) -> (100, 1, 128)
                    src_pcd,                                # (947, 1, 128)
                    # (8, 100, 947) head 수(8)만큼 복붙, multi-head attetion 할라고
                    memory_mask=batched_attn.repeat_interleave(self.num_heads, dim=0).permute((0, 2, 1)),
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=batched_pos_enc.permute((1, 0, 2)),
                    query_pos=query_pos
                )

                ################### songoh mask PE refinement part ###################
                # decomposed_coord = []
                # decomposed_coord.append(coords[hlevel].decomposed_features)
                # decomposed_coord = decomposed_coord[0]
                #
                # batched_coord = torch.stack([
                #     decomposed_coord[k][rand_idx[k], :] for k in range(len(rand_idx))
                # ])
                # songohmask = batched_attn.permute(0, 2, 1)
                # list_dict = {}
                # avg = []
                # # print(len(songohmask))
                # if torch.all(songohmask) or torch.all(~songohmask):
                #     pass
                # else:
                #     for song in range(len(songohmask)):
                #         list_dict[f"idx_song{song}"] = []
                #         for s in range(len(songohmask[song])):
                #             list_dict[f"idx_song{song}"].append(torch.nonzero(songohmask[song][s]))
                #
                #             # print('debut msg')
                #     # print('debut msg')
                #     for count in range(len(songohmask)):
                #         for s in range(len(songohmask[count])):
                #             selected_elements = batched_coord[count][list_dict[f"idx_song{count}"][s]]
                #             selected_elements = selected_elements.unsqueeze(1)
                #             average = torch.mean(selected_elements, dim=0)
                #             # if torch.isnan(average).all():
                #             #     nan = average
                #             #     print("naaaaaaaaaaaaaaaaan")
                #             avg.append(average)
                #     avg_result = torch.cat(avg, 0).reshape(len(songohmask), -1, 3)
                #     avg_result = torch.where(torch.isnan(avg_result), sampled_coords, avg_result)
                #     query_pos = self.pos_enc(avg_result.float(),
                #                              input_range=[mins, maxs]
                #                              )  # Batch, Dim, queries
                #     query_pos = self.query_projection(query_pos)
                #     query_pos = query_pos.permute((2, 0, 1))
                #     # print('debut msg')
                ################### songoh mask PE refinement part ###################

                # 정제된 쿼리들 간의 self attention
                # self attention in Query Refinement
                output = self.self_attention[decoder_counter][i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_pos
                )

                # FFN
                queries = self.ffn_attention[decoder_counter][i](
                    output
                ).permute((1, 0, 2))

                # Position Refinement -SongOh3
                # Query Content 기반의 Position Refinement #2
                if self.pos_refinement is not None:
                    if self.pos_embed_differ_each_layer:
                        tmp = self.pos_refinement[hlevel](queries)
                    else:
                        tmp = self.pos_refinement(queries)
                    # refine V1
                    # tmp[..., :3] += inverse_sigmoid(sigmoid(sampled_coords))
                    # refine V2
                    # check1 = inverse_sigmoid(obj_anchor.permute(1, 0, 2))

                    #### 6D Anchor ####
                    # tmp[..., :self.anchor_dim] += inverse_sigmoid(obj_anchor.permute(1, 0, 2))
                    # new_reference_points = tmp[..., :self.anchor_dim].sigmoid()
                    #### 6D Anchor ####

                    #### 3D Coord ####
                    # tmp[..., :3] += inverse_sigmoid(obj_anchor.permute(1, 0, 2))
                    # tmp[..., :3] += inverse_sigmoid(obj_anchor)
                    # new_reference_points = tmp[..., :3].sigmoid()

                    ####### No need to Sigmoid -0824 songoh #########
                    # tmp[..., :3] += obj_anchor
                    obj_anchor += tmp[..., :3]
                    # new_reference_points = tmp[..., :3]
                    new_reference_points = obj_anchor
                    #### 3D Coord ####

                    # reference_points = new_reference_points.detach().permute((1, 0, 2))
                    reference_points = new_reference_points.detach()
                    # query_pos = self.pos_enc(sampled_coords.float(),
                    #                          input_range=[mins, maxs]
                    #                          )  # Batch, Dim, queries

                    # query_pos = self.query_projection(reference_points)

                    #### 6D Anchor ####
                    # query_sine_embed = self.gen_sineembed_for_position(reference_points).permute((1, 2, 0))
                    #### 6D Anchor ####

                    #### 3D Coord ####
                    # query_sine_embed = self.pos_enc(reference_points, input_range=[mins, maxs]).permute((1, 2, 0))
                    # query_sine_embed = self.pos_enc(reference_points, input_range=[mins, maxs]).permute((1, 2, 0))
                    query_sine_embed = self.pos_enc(reference_points.float(), input_range=[mins, maxs])
                    #### 3D Coord ####

                    query_pos = self.query_projection(query_sine_embed).permute((2, 0, 1))
                    # query_sine_embed = query_sine_embed.permute((2, 0, 1))
                    # query_pos = query_pos.permute((2, 0, 1))
                    # query_pos = query_pos.permute((2, 0, 1))
                    # print('check')
                # if self.training:
                #     # dn_query, dn_reference = self.make_dn_query(queries, obj_anchor, mask_dict)
                #     # out_boxes = self.pred_box(dn_reference, dn_query)
                #     out_boxes = self.pred_box(obj_anchor, queries)
                #     prediction_bbox.append(out_boxes)

                predictions_class.append(output_class)
                predictions_mask.append(outputs_mask)

                # MM + Query Refinement loop End
        # if self.training:
        #     # dn_query, dn_reference = self.make_dn_query(queries,obj_anchor, mask_dict)
        #     # out_boxes = self.pred_box(dn_reference, dn_query)
        #     out_boxes = self.pred_box(obj_anchor, queries)
        #     prediction_bbox.append(out_boxes)
            # print('check')
        if self.train_on_segments:      # False > True
            # 정제 과정을 마친 쿼리로 마스크 예측
            output_class, outputs_mask = self.mask_module(queries,
                                                          mask_features,
                                                          mask_segments,
                                                          0,
                                                          ret_attn_mask=False,
                                                          point2segment=point2segment,
                                                          coords=coords)

        # Query Refinement -> queries -> mask module
        else:
            # 정제 과정을 마친 쿼리로 마스크 예측
            output_class, outputs_mask = self.mask_module(queries,
                                                          mask_features,
                                                          None,
                                                          0,
                                                          ret_attn_mask=False,
                                                          point2segment=None,
                                                          coords=coords)
        # append at last of list
        if self.training:
            # print('dn post process')
            if mask_dict is not None:
                output_class, outputs_mask = self.dn_post_process(output_class, mask_dict, outputs_mask)
        predictions_class.append(output_class)
        predictions_mask.append(outputs_mask)

        # 최종 예측된  class, mask는 main loss 할당
        # 나머지 중간 계층 예측 결과물은 aux loss 할당
        # denosing query에서 예측된 결과물만 잘라서
        # dn loss 계산되도록 만들면 됨 -0807 sonoh
        if self.training:
            return {
                'pred_logits': predictions_class[-1],
                'pred_masks': predictions_mask[-1],
                # 'pred_boxes': prediction_bbox[-1],
                # 'aux_outputs': self._set_train_aux_loss(
                #     predictions_class, predictions_mask, prediction_bbox
                # ),
                'aux_outputs': self._set_aux_loss(
                    predictions_class, predictions_mask
                ),
                'sampled_coords': sampled_coords.detach().cpu().numpy() if sampled_coords is not None else None,
                'backbone_features': pcd_features
            }
        else:
            return {
                'pred_logits': predictions_class[-1],
                'pred_masks': predictions_mask[-1],
                'aux_outputs': self._set_aux_loss(
                    predictions_class, predictions_mask
                ),
                'sampled_coords': sampled_coords.detach().cpu().numpy() if sampled_coords is not None else None,
                'backbone_features': pcd_features
            }
    # MM
    def mask_module(self, query_feat, mask_features, mask_segments, num_pooling_steps, ret_attn_mask=True,
                    point2segment=None, coords=None):
        query_feat = self.decoder_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)
        outputs_class = self.class_embed_head(query_feat)

        output_masks = []

        if point2segment is not None:
            output_segments = []
            for i in range(len(mask_segments)):
                output_segments.append(mask_segments[i] @ mask_embed[i].T)
                output_masks.append(output_segments[-1][point2segment[i]])
        else:  # True
            for i in range(mask_features.C[-1, 0] + 1):
                # Point Feature F0 & Instance Feature dot product
                # mask features is Point Feautres
                # mask embed is Instance Feature
                output_masks.append(mask_features.decomposed_features[i] @ mask_embed[i].T)

        output_masks = torch.cat(output_masks)
        outputs_mask = me.SparseTensor(features=output_masks,
                                       coordinate_manager=mask_features.coordinate_manager,
                                       coordinate_map_key=mask_features.coordinate_map_key)

        if ret_attn_mask:  # True
            attn_mask = outputs_mask
            # PointFeature Number에 맞게 pooling
            for _ in range(num_pooling_steps):
                attn_mask = self.pooling(attn_mask.float())

            # check1 = attn_mask.F.detach().sigmoid()
            # check2 = (attn_mask.F.detach().sigmoid() < 0.5)
            # songoh_check = attn_mask.F.detach().sigmoid()
            # check3 = attn_mask.F.detach().sigmoid()
            # sigmoid() > 0.5 -> True, sigmoid() < 0.5 -> False
            attn_mask = me.SparseTensor(features=(attn_mask.F.detach().sigmoid() < 0.5),
                                        coordinate_manager=attn_mask.coordinate_manager,
                                        coordinate_map_key=attn_mask.coordinate_map_key)

            if point2segment is not None:
                return outputs_class, output_segments, attn_mask
            else:
                return outputs_class, outputs_mask.decomposed_features, attn_mask

        if point2segment is not None:  # None
            return outputs_class, output_segments
        else:
            return outputs_class, outputs_mask.decomposed_features

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]

    @torch.jit.unused
    def _set_train_aux_loss(self, outputs_class, outputs_seg_masks, outputs_boxes):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b, "pred_boxes": c}
            for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_boxes[:-1])
        ]


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        self.orig_ch = channels
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor, input_range=None):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        pos_x, pos_y, pos_z = tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]
        sin_inp_x = torch.einsum("bi,j->bij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("bi,j->bij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("bi,j->bij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)

        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)

        emb = torch.cat((emb_x, emb_y, emb_z), dim=-1)
        return emb[:, :, :self.orig_ch].permute((0, 2, 1))

# MSA Songoh
class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask = None,
                     tgt_key_padding_mask= None,
                     query_pos = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask= None,
                    tgt_key_padding_mask = None,
                    query_pos = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask = None,
                tgt_key_padding_mask = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)

# CrossAttetion Songoh
class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask = None,
                     memory_key_padding_mask = None,
                     pos = None,
                     query_pos = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask = None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None):
        tgt2 = self.norm(tgt)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)
#MLP Songoh
class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


# Songoh MLP for position refinement
class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def sigmoid(x):
   """Sigmoid Activation Function
      Arguments:
      x.torch.tensor
      Returns
      Sigmoid(x.torch.tensor)
   """
   return 1 / (1+torch.exp(x))
# def sigmoid(x):
#     x = x.cpu().numpy()
#     result = 1 / (1 +np.exp(-x))
#     return result





