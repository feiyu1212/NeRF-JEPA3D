import os
import glob
import shutil
import configargparse
import tqdm
import imageio
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as T

from models.render_image import render_single_image
from models.model import VisionNerfModel
from models.sample_ray import RaySamplerSingleImage
from models.projection import Projector
from utils import img_HWC2CHW

def config_parser():
    parser = configargparse.ArgumentParser()
    # general
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--expname', type=str, help='experiment name')
    parser.add_argument('--ckptdir', type=str, help='checkpoint folder')
    parser.add_argument('--ckpt_path', type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument('--outdir', type=str, help='output video directory')
    parser.add_argument("--local_rank", type=int, default=0, help='rank for distributed training')
    parser.add_argument("--include_src", action="store_true", help="Include source views in calculation")

    ########## dataset options ##########
    ## render dataset
    parser.add_argument('--data_path', type=str, help='the dataset to train')
    parser.add_argument('--data_type', type=str, default='srn', help='dataset type to use')
    parser.add_argument('--img_hw', type=int, nargs='+', help='image size option for dataset')
    parser.add_argument('--data_range', type=int,
                        default=[0, 50],
                        nargs='+',
                        help='data index to select from the dataset')
    parser.add_argument('--data_indices', type=int,
                        default=[0],
                        nargs='+',
                        help='data index to select from the dataset')
    parser.add_argument('--use_data_index', action='store_true',
                        help='use data_indices instead of data_range')
    parser.add_argument('--pose_index', type=int,
                        default=64,
                        help='source pose index to select from the dataset')
    parser.add_argument('--source_view_list', type=str, default="",
                        help='path to source view list, overrides pose_index if not empty')
    parser.add_argument('--no_reload', action='store_true',
                        help='do not reload weights from saved ckpt (not used)')
    parser.add_argument('--distributed', action='store_true', help='if use distributed training (not used)')
    parser.add_argument('--skip', type=int,
                        default=1,
                        help='camera pose skip')
    parser.add_argument("--multicat", action="store_true",
                        help="Prepend category id to object id. Specify if model fits multiple categories.")

    ########## model options ##########
    ## ray sampling options
    parser.add_argument('--chunk_size', type=int, default=128,
                        help='number of rays processed in parallel, decrease if running out of memory')
    
    ## model options
    parser.add_argument('--im_feat_dim', type=int, default=128, help='image feature dimension')
    parser.add_argument('--mlp_feat_dim', type=int, default=512, help='mlp hidden dimension')
    parser.add_argument('--freq_num', type=int, default=10, help='how many frequency bases for positional encodings')
    parser.add_argument('--mlp_block_num', type=int, default=2, help='how many resnet blocks for coarse network')
    parser.add_argument('--coarse_only', action='store_true', help='use coarse network only')
    parser.add_argument("--anti_alias_pooling", type=int, default=1, help='if use anti-alias pooling')
    parser.add_argument('--num_source_views', type=int, default=1, help='number of views')
    parser.add_argument('--freeze_pos_embed', action='store_true', help='freeze positional embeddings')
    parser.add_argument('--no_skip_conv', action='store_true', help='disable skip convolution')

    ########### iterations & learning rate options (not used) ##########
    parser.add_argument('--lrate_feature', type=float, default=1e-3, help='learning rate for feature extractor')
    parser.add_argument('--lrate_mlp', type=float, default=5e-4, help='learning rate for mlp')
    parser.add_argument('--lrate_decay_factor', type=float, default=0.5,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument('--lrate_decay_steps', type=int, default=50000,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument('--warmup_steps', type=int, default=10000, help='num of iterations for warm-up')
    parser.add_argument('--scheduler', type=str, default='steplr', help='scheduler type to use [steplr]')
    parser.add_argument('--use_warmup', action='store_true', help='use warm-up scheduler')
    parser.add_argument('--bbox_steps', type=int, default=100000, help='iterations to use bbox sampling')

    ########## rendering options ##########
    parser.add_argument('--N_samples', type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument('--N_importance', type=int, default=64, help='number of important samples per ray')
    parser.add_argument('--inv_uniform', action='store_true',
                        help='if True, will uniformly sample inverse depths')
    parser.add_argument('--det', action='store_true', help='deterministic sampling for coarse and fine samples')
    parser.add_argument('--white_bkgd', action='store_true',
                        help='apply the trick to avoid fitting to white background')

    parser.add_argument('--x', type=int, default=0, help='x')

    return parser

def parse_pose_dvr(path, num_views):
    cameras = np.load(path)

    intrinsics = []
    c2w_mats = []

    for i in range(num_views):
        # ShapeNet
        wmat_inv_key = "world_mat_inv_" + str(i)
        wmat_key = "world_mat_" + str(i)
        kmat_key = "camera_mat_" + str(i)
        if wmat_inv_key in cameras:
            c2w_mat = cameras[wmat_inv_key]
        else:
            w2c_mat = cameras[wmat_key]
            if w2c_mat.shape[0] == 3:
                w2c_mat = np.vstack((w2c_mat, np.array([0, 0, 0, 1])))
            c2w_mat = np.linalg.inv(w2c_mat)

        intrinsics.append(cameras[kmat_key])
        c2w_mats.append(c2w_mat)

    intrinsics = np.stack(intrinsics, 0)
    c2w_mats = np.stack(c2w_mats, 0)

    return intrinsics, c2w_mats

class DVRRenderDataset(Dataset):
    """
    Dataset for rendering
    """
    def __init__(self, args, mode="test", **kwargs):
        """
        Args:
            args.data_path: path to data directory
            args.img_hw: image size (resize if needed)
        """
        super().__init__()
        self.base_path = args.data_path
        self.dataset_name = os.path.basename(args.data_path)
        assert os.path.exists(self.base_path)

        cats = [x for x in glob.glob(os.path.join(args.data_path, "*")) if os.path.isdir(x)]

        list_prefix = "softras_"

        if mode == "train":
            file_lists = [os.path.join(x, list_prefix + "train.lst") for x in cats]
        elif mode == "val":
            file_lists = [os.path.join(x, list_prefix + "val.lst") for x in cats]
        elif mode == "test":
            file_lists = [os.path.join(x, list_prefix + "test.lst") for x in cats]

        print("Loading NMR dataset", self.base_path, "name:", self.dataset_name, "mode:", mode)

        self.mode = mode

        all_objs = []
        for file_list in file_lists:
            if not os.path.exists(file_list):
                continue
            base_dir = os.path.dirname(file_list)
            cat = os.path.basename(base_dir)
            with open(file_list, "r") as f:
                objs = [(cat, os.path.join(base_dir, x.strip())) for x in f.readlines()]
            all_objs.extend(objs)

        self.all_objs = all_objs

        # self.all_objs = self.all_objs[:100] # HACK to skip all other dataset

        self.intrinsics = []
        self.poses = []
        self.rgb_paths = []
        for _, path in tqdm.tqdm(self.all_objs):
            curr_paths = sorted(glob.glob(os.path.join(path, "image", "*")))
            self.rgb_paths.append(curr_paths)

            pose_path = os.path.join(path, 'cameras.npz')
            intrinsics, c2w_mats = parse_pose_dvr(pose_path, len(curr_paths))

            self.poses.append(c2w_mats)
            self.intrinsics.append(intrinsics)

        self.rgb_paths = np.array(self.rgb_paths)
        self.poses = np.stack(self.poses, 0)
        self.intrinsics = np.array(self.intrinsics)

        assert(len(self.rgb_paths) == len(self.poses))

        self.define_transforms()
        self.img_hw = args.img_hw

        # default near/far plane depth
        self.z_near = 1.2
        self.z_far = 4.0

    def __len__(self):
        return len(self.intrinsics)

    def define_transforms(self):
        self.img_transforms = T.Compose(
            [T.ToTensor(), T.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))]
        )
        self.mask_transforms = T.Compose(
            [T.ToTensor(), T.Normalize((0.0,), (1.0,))]
        )

    def __getitem__(self, index):
        # Read source RGB
        src_rgb_paths = self.rgb_paths[index]
        src_c2w_mats = self.poses[index]
        src_intrinsics = self.intrinsics[index].copy()

        src_rgbs = []
        src_masks = []
        for i, rgb_path in enumerate(src_rgb_paths):
            img = imageio.imread(rgb_path)[..., :3]
            mask = (img.sum(axis=-1) != 255*3)[..., None].astype(np.uint8) * 255
            rgb = self.img_transforms(img)
            mask = self.mask_transforms(mask)

            src_intrinsics[i, 0, 0] *= img.shape[1] / 2.0
            src_intrinsics[i, 1, 1] *= img.shape[0] / 2.0
            src_intrinsics[i, 0, 2] = img.shape[1] / 2.0
            src_intrinsics[i, 1, 2] = img.shape[0] / 2.0

            h, w = rgb.shape[-2:]
            if (h != self.img_hw[0]) or (w != self.img_hw[1]):
                scale = self.img_hw[-1] / w
                src_intrinsics[i, :2] *= scale

                rgb = F.interpolate(rgb[None, :], size=self.img_hw, mode="area")[0]
                mask = F.interpolate(mask[None, :], size=self.img_hw, mode="area")[0]
            
            src_rgbs.append(rgb)
            src_masks.append(mask)

        depth_range = np.array([self.z_near, self.z_far])

        return {
            "rgb_path": rgb_path,
            "img_id": index,
            "img_hw": self.img_hw,
            "src_masks": torch.stack(src_masks).permute([0, 2, 3, 1]).float(),
            "src_rgbs": torch.stack(src_rgbs).permute([0, 2, 3, 1]).float(),
            "src_c2w_mats": torch.FloatTensor(src_c2w_mats),
            "src_intrinsics": torch.FloatTensor(src_intrinsics),
            "depth_range": torch.FloatTensor(depth_range)
        }

def gen_eval(args):

    device = "cuda"
    print(f"checkpoints reload from {args.ckptdir}")

    dataset = DVRRenderDataset(args)

    # Create VisionNeRF model
    model = VisionNerfModel(args, False, False)
    # create projector
    projector = Projector(device=device)
    model.switch_to_eval()

    if args.use_data_index:
        data_index = args.data_indices
    else:
        data_index = np.arange(args.data_range[0], args.data_range[1])

    use_source_list = len(args.source_view_list) > 0
    if use_source_list:
        print("Using views from list", args.source_view_list)
        with open(args.source_view_list, "r") as f:
            tmp = [x.strip().split() for x in f.readlines()]
        pose_indices = {
            x[0] + "/" + x[1]: torch.tensor(list(map(int, x[2:])), dtype=torch.long)
            for x in tmp
        }
    else:
        pose_indices = [args.pose_index]

    for d_idx in data_index:      
        dpath = dataset[d_idx]["rgb_path"][:-15]
        obj_basename = os.path.basename(dpath)
        cat_name = os.path.basename(os.path.dirname(dpath))
        obj_name = cat_name + "_" + obj_basename if args.multicat else obj_basename
        if obj_name not in ['02691156_4bb41171f7e6505bc32f927674bfca67','02933112_9b28e4f5c66daae45f29768b6bb620cb','03001627_34898c36e711fbde713decb1a0563b12', '03691459_a1d1f232168607c81dd4da6e97c175c2','04256520_5cc1b0be0eb9a6085dca6305fb9f97ca','02691156_4c5b8babafbb7b5f937ae00fead8910d','02933112_9bcfb450ed3046a74da5feafe6f1c8fc','03001627_34b1b2ee54ea314486a1cb4b695edbd9','03691459_a3f14846404245d5bbbcb091dd094e5f','04256520_5d2ff1a62e9d58baeaf288f952624966','02691156_4d0898c7cac1199a4b0853084d4042f7','02933112_9c802e8971c1b25f480fa521a9c7198a','03001627_35053caa62eea36c116cc4e115d5fd2','03691459_a551777c3b35c06f43f491729931f44','04256520_5d9f1c6f9ce9333994c6d0877753424f','02691156_4d2d87e61a18760ff64801ad2940cdd5','02933112_9cec9c8b65ec9e4f4c62bee40dcdc539','03001627_3526528270d5f3f766268502c798b9af','03691459_a9957cf39fdd61fc612f7163ca95602','04256520_5e6fe3ce7afd4490441e918a50adf356','02691156_4de5861211ad5b95e7ef9fff09638f8e','02933112_9dfac0132596ff09b13b0af4c7a59aa0','03001627_357275196334dc4feaf288f952624966','03691459_a99cc4f1326b9b48b08cd968d6703683','04256520_5eae999a81861da03bec56bff764ba78','02691156_4e4ae13865bf47f41adbb4c4e06ad649','02933112_9e6434ba1ad59fa611fc2b865c2a185b','03001627_359331da71ed26eca6c03a53cf0a14c9','03691459_ab5c8c38528f06daf398d0c5832df00e','04379243_26ab5349a902d570d42b9650f19dd425','02691156_4e67529b0ca7bd4fb3f2b01de37c0b29','02933112_9f07ee5bcf6a4b11151b305e5243d9f8','03001627_35bcb52fea44850bb97ad864945165a1','03691459_ab651261126de20c145adb610a878e88','04379243_272a4cf3cfff3eb1e173cee47fbaa88','02691156_4ea714b785e24f0d9a6e43b878d5b335','02933112_9f17f45a28d063e7391e4d6c585a697a','03001627_35c2de57ee36f652492d9da2668ec34c','03691459_ad2e92448857e1cc6ad08aa387990063','04379243_27805445a3c587c1db039d8689a74349','02691156_4eced94670d10b35e856faf938562bd0','02933112_9fe8af4bc8683818579c9a60a6b30a2a','03001627_35e77eed59e1113c22e4620665c23c97','03691459_add914368a6ca448732bda87f2718525','04379243_279351131d06222cbe9bca6d7b2e5b3','02691156_4ee48907120e261c3df16180af9974ee','02933112_a03797c4034be11ac59350d819542ec7','03001627_35ee4bcad88ab50af6e44a01c524295b','03691459_ae4bcb4baca763c24521562865ab775e','04379243_27a90972dfe64be5c3bd24f986301745','02691156_4f2830d3d566220be5dd38f670a033a8','02933112_a07f3b931238a5c3fe618e12c7f65698','03001627_35f83268d4280532dc89a28b5e6678e0','03691459_aed2ee05cf37c85c9a8c31231dd99d82','04379243_27f9965a337bf46d85924458b86f34','02691156_4f3a64164fbe16f54c2c88971423d0be','02933112_a08aa6e386cd983c59350d819542ec7','03001627_367dc1e6752cabbcc34bba142e6e15e6','03691459_aed97e60cd2802ce7ffb47acd56f396b','04379243_28001cb70c38f19cf32b6091d9628440','02691156_4f3f39ddde5874f2db73445864b72015','02933112_a0beaab9a8c2f0954e7d60def15dcb8b','03211117_e7b9c172059d23a6f12f3a2695789ca4','03691459_afe96f3cf256cbac81a6b6721af23c58','04379243_281f296380a0e4a81db7abc68608fde1','02691156_4fb10ce02887e35efca1f1143bb6bc17','02933112_a0eb46b125a99e26473aef508bd8614e','03211117_e912fab93eb63fef6b4897a481d7d16a','03691459_b0ba01e0a7f3553897950841baebd2bd','04379243_282d36d7ca9705f6ca421e9e01647b4a','02691156_4fd9c86e43a1dea17209009cfb89d4bd','02933112_a3a6f9e80956ec6f4035f93ab9531db','03211117_e9466e872848075d3aeab48ed64c0fa4','04090263_11126129151f3e36afb1ffba52bacfa2','04379243_285857e7d12f1b74a4d2a71d4ca57f99','02691156_4fe076aa34c706b83d7edb3bb2d24b58','02958343_2b236f0333fe789ed4b0a2774191078f','03211117_ea4b90cca359e2d38b9be827bf6fc77','04090263_12038871e583f645af56497f051566ab','04379243_28912046d42946df7db48c721db3fba4','02691156_50755e616df58fe566cf1b4a8fc3914e','02958343_2b7101c3a9c9aea533116515e458cb86','03211117_ea7dc70f0ef4b04bcbe0b8af4aec5d6c','04090263_1345ba2c3b27ba786bb9681d4604e123','04379243_28e64eefcada205fddf120185fc0b2a0','02691156_52185f504ffa9b32ca8607f540cc62ba','02958343_2b711c735df95aded8df13fb69e08d76','03211117_ecb3d57cc0e8018f3f6b4923416758fd','04090263_138cbc9b5456cfef55d33831e71fa52','04379243_28f3844a7a3d37399c0ca344f487323e','02691156_525446bc8f55e629151f2df200a24ac','02958343_2b72fb5d7ce07564f961db66e477976d','03211117_ed31aec87c045ebdebe17c8dfb911d2c','04090263_13ff0ba1e8d79a927f71da36e5c530dc','04379243_28f702b5c6ccffe7fcf9154182ccb5a4','02691156_52712e1c07ea494419ba010ddb4974fe','02958343_2b766a7abde647c7175d829215e53daf','03211117_f1a3e41b45de104a810988cb5fefedde','04090263_14139593048f806e79093d8935cfe4f0','04379243_28fb9a81898f88c4ae8375def5e736d8','02691156_52747021197c7eeeb1a0518c224975f','02958343_2b8c1b23b617671d1a964dea8a7aab','03211117_f1d77e0f4a2adc2bb305a938e0ed1b48','04090263_1632f8ce3254cfccc8c51c73cb7275ed','04379243_29207ae4e01c415894fc399eb2c2f348','02691156_52a1b6e8177805cf53a728ba6e36dfae','02958343_2b92417cee9dfe0cb94d1aa50e685ffa','03211117_f240248beae8d20661049a5d9182333f','04090263_164248eefde5ce846314c3b6a3a65519','04379243_2927b65bc7b9561bf51f77a6d7299806','02691156_52a84fea7c314f4c3dfc741b4df74043','02958343_2ba7ea78b9b6e2a6b420256a03f575c0','03211117_f3d4cb310a6106f5e66202687a227eab','04090263_739971469c9903eeb6dc6c452bb50aac','04379243_29b55c6cc05c37606e066e11deddab75','02691156_52c9b1a9f8d3cbcb9a6e43b878d5b335','02958343_2be8bd6b150a9fee97df6383472cc5b6','03211117_f4877a34163978f84efc7e7114f1b9c5','04090263_73ce9a5617acc6dbf1e0fcef68407ae5','04379243_29c6a184dfee3050820018801b237b3d','02691156_52cd5876945106d154eefcdc602d4520','02958343_2bf9ed80cb75411f58dbdf2b5c6acfca','03211117_f6515d1343e25bab8913de0e5cfdcafb','04090263_73e0ab2e1d0ea752bc6e84bc30e573cd','04379243_29def96b77d4cfe372f9a9353d57f9ef','02691156_52e7f93d592622a9615ba7bf3620290d','02958343_2c6b14bcd5a5546d6a2992e9465c023b','03211117_f800fbcdef1ac9038b5dbcd4be5ceef8','04090263_74e930c54ddaf8add34ad43a472bf958','04379243_2a0eff86efdf60a89a682a6d0e92c30','02691156_530540dc968b396d7f3805c1aec66f00','02958343_2ca59da312d06381b927782fc69a1fbb','03211117_f84f6c14852ddad1d06e6be4f8954ac','04090263_7537fb364b25e32e999562d1784e5908','04379243_2a896f1d6925cc61dc89a28b5e6678e0','02691156_53edcc6832e776dcca8607f540cc62ba','02958343_2cc4573bb9441493d12568d09c2fba02','03211117_fc314f1e3325a47af287ec53a469521','04090263_76377dc3d6b1dad2c0aaedf10d776af7','04379243_2aad9a8f3db3272b916f241993360686','02691156_53f0e2f6671346ae5ff3feb917a6004b','02958343_2cdf160f331b2f402f732d6dce34e9b3','03211117_fc542f42b786ae20c40162916d18053d','04090263_76dd6c8a2cf09887bbb7f70943d3cb52','04379243_2ab09f4db5f5e842bf595b60a303303','02691156_543412ccea0db2f8f37f38dedb2f1219','02958343_2ce3965eb931e7c1efdff89bf9a96890','03211117_fdf3953c665c36fdeb47c06d7d8c2d65','04090263_77241daf76a045c099d9d900afe054b8','04379243_2ab79a94145330a95ca21a5844017a0f','02828884_895563d304772f50ad5067eac75a07f7','02958343_2d031e07ae160bcfdd141480e2c154d3','03636649_7a8615c643bc3d96ed6eef8e856a36ea','04090263_7744efae453f26c05e9263096a26104d','04379243_2abe61af67cbd99aaa1d46a2befc5e09','02828884_89e2eaeb437cd42f85e40cb3507a0145','02958343_2d096d5dc0e0eeb370a43c2d978e502e','03636649_7b39100755e9578799284d844aba7576','04090263_7787bf25c9417c4c31f639996cb3d35d','04379243_2ae89daf7433f4d14b3c42e318f3affc','02828884_8a5a59ab999c03ccfb0eb7e753c06942','02958343_2d1718bda87f59dc673ddeabdcc8c6e','03636649_7bc1b202ebf000625949e084b65603cf','04090263_7816689726ed8bbfc92b24247435700c','04379243_2b06a917abc1150b554ad4a156f6b68','02828884_8a6f07f3d357fbfd2b12aa6a0f050b3','02958343_2d1adb247cc7a1e353da660a6567c5ff','03636649_7be01530bf43f2ed8a83637b92bdc7','04090263_78a0c4ff75258ecf16b34c3751bc447d','04379243_2b0c16b26ebfb88f490ad276cd2af3a4','02828884_8aabc6c97aeacae7ad5067eac75a07f7','02958343_2d5c34ee6afae94a23c74c52752d514f','03636649_7bebdd742342ba93febad4f49b26ec52','04090263_78dd4dc65639258cd735fa1ab17311ec','04379243_2b1c1e0fad3cb3b8fad46760e869d184','02828884_8b98dbc11c5d2fb7601104cd2d998272','02958343_2d5c6f88dc81b668283ffcfc40c29975','03636649_7c23362b39f318cbb18d6f615cb18bdd','04090263_78e2a75ff1d93138e8914057d776d90b','04401088_eed72e5bc3dc4e85150c05906b260c9e','02828884_8d1f361eb7a927d8907921e9162f6a43','02958343_2d730665526ee755a134736201a79843','03636649_7eadde33d9f9d8b272e526c4f21dfca4','04090263_79461905bb97ebc21a77229b9f90bf5','04401088_f18dbf3cbc7d3822de764ca2f457c756','02828884_8d218bceb517f272f155d75bbf62b80','02958343_2dbd3e61fe27cd29bddac102ee6384','03636649_8006e3cb3ba934a05b977412e02c412c','04090263_79f507c3befe69ba2987a7b722c00b7d','04401088_f46531484dea3574a803a040655859ad','02828884_8f52743c3cb564f149e6f7df978f3373','02958343_2dbfc333297696121ca2e8373597f711','03636649_80436dff2a30721849655ac7c771b113','04090263_7a79d63c2cf6df519f605c8c86eb1ec2','04401088_f649a5368586aa77eecb4e0df212ad9','02828884_8f64075972a1f7f3dc18af6f6bfce3ef','02958343_2dbfe9c041fc6b3a94ce9d7b04676231','03636649_809ed642119aa5799e8bf807e902261','04090263_7aff2693a24b4faed957eaf7f4edb205','04401088_f75cac5b112f14002c32dcd0becbedb7','02828884_9027bc9728f006bb40f0ac0fb9a650d','02958343_2dcc5ed5780462da124dae0a32996c4c','03636649_80a5ce76688700a2fdd36147bc6b8031','04090263_7b1b02b8f71226e4bb9224760a70fece','04401088_f77811bd35b9215dfd06b6d6f44d62dc','02828884_90a309f7ac2c947f155d75bbf62b80','03001627_108238b535eb293cd79b19c7c4f0e293','03636649_81bbe86e17196c2fd0db1e44373ee2d4','04090263_7bd6db14ec6c37efeac2c9f41d1276bf','04401088_f928f74ed34e46c4b5ce02cb8ffbdc86','02828884_90a8b80aa2fa209ca936e2693dce34b6','03001627_108b9cb292fd811cf51f77a6d7299806','03636649_8508808961d5a0b2b1f2a89349f43b2','04090263_7bfdd659a04c412efa9286f039319ff7','04401088_fa14355e4b5455d870f7cd075189ddd8','02828884_916a6ff06a128851ed98cca8f0ccd5f7','03001627_10dc303144fe5d668d1b9a1d97e2846','03636649_8522fb13bfa443d33cabd62faf4bd0f0','04090263_7c31ae88ca4faa649a2ee232a197081e','04401088_fbd120d2c01484d56c95c6d882af3c0','02828884_919f90e92283eff8febad4f49b26ec52','03001627_11040f463a3895019fb4103277a6b93','03636649_85574f6036ea1f90d8c46a3a266762d7','04090263_7c426a52358e8c1d64c4db7c3b428292','04401088_fe39a57b8137ecbd5b2233351507f22f','02828884_91e169ea3ceb587beff42b9e13c388bc','03001627_111cb08c8121b8411749672386e0b711','03636649_85a73c46a97649fa6d0c88a73d7cb14d','04090263_7c6a21d8b91fd12a1b7837f7a64e3031','04401088_ff1e484e02e001bdf8a0d9b8943b4050','02828884_92f1fa8d3b5da497ad5067eac75a07f7','03001627_11347c7e8bc5881775907ca70d2973a4','03636649_85f8a8c585742c9b96a3517f50eeb9f4','04090263_7d310ff81ee66564b38e8b1e877a5704','04530566_9004946f75082a8632c0857fb4bcf47a','02828884_943dde2754ddc7822e8ff3556a90169','03001627_114f72b38dcabdf0823f29d871e57676','03636649_86d556273aa5075aaa660c42e675c161','04090263_7dba6294173994131226a6f096e4f8c8','04530566_90bf73b91185303139555c8c231d0eb7','02828884_95b375118d800de7ad5067eac75a07f7','03001627_11740d372308f12185047f9f654ddc2e','03636649_87107afb7ad115414b3c42e318f3affc','04090263_7e316474bc8b072fca74c4e4ab012aef','04530566_90d83e1dde32426407e66c6e74f5ce3','02828884_95eed587c3728d22601104cd2d998272','03001627_117bd6da01905949a81116f5456ee312','03636649_873aad4a222d541a91c2792fcdc1ca8','04090263_7e5c0d35215be21ff3998727b15249db','04530566_90e6c6083fcd47833e45dd2f173cbf9b','02828884_9699995246fd521ca909cd1ba5751669','03001627_1190af00b6c86c99c3bd24f986301745','03636649_882aae761370df14786810c22b062a88','04256520_52d307203aefd6bf366971e8a2cdf120','04530566_9115745cde6f30c67f141c9abbacbdb8','02828884_976636850849c3d6ffd996335233167','03001627_11c9c57efad0b5ec297936c81e7f6629','03636649_886ff4f5cd90c6ad39b3360f500ac52a','04256520_52dd0fac460adb45e2879d5d9f05633','04530566_91a124454518abb7f2ad837508eb2db7','02828884_9787c8521ba8d46b5b83c5170da4c6c2','03001627_11d4f2a09184ec972b9f810ad7f5cbd2','03636649_8872dceb7ba9f34c140406de8e63ea3a','04256520_536cae63d37ef32265ba78ad9601cf1b','04530566_91e0e1a6dbf302c3d55da98ad008849b','02828884_9790980a8ff823287ed9296ee19fa384','03001627_11e28120789c20abc8687ff9b0b4e4ac','03636649_892900c66731d9c473ab7b7128d466a2','04256520_541e331334c95e5a3d2617f9171b5ccb','04530566_921a5d88994aa99c71327f667b2179b0','02828884_98971eb747738a6880360680c1602c7d','03001627_11e6e58798ae5be83e5b5dbd84cdd0f8','03636649_8937a2d361775c68aafd61baec633e88','04256520_54a209955f7a47bed8e8a8a207ee5bd2','04530566_925c05dbefada808cfe472915a175bb','02828884_9897a75c64a369d458c73770090b865','03001627_128517f2992b6fb92057e1ae1c4cc928','03636649_896abd405c79547086485c798787f66b','04256520_5560a425c597eacbff841fc99bb16039','04530566_9262aa413df7b369d735fa1ab17311ec','02828884_990c56d6ab64279c2056b4bd5d870b47','03001627_12f395270a3316d01666e1246e760f82','03636649_89ed63af13b79f3ef42a90fe4baf4591','04256520_55e0dfba8cc226871b17743c18fb63dc','04530566_92e4ae4dfff684832dbef90d406185fa','02828884_991803aca7fca258b40f0ac0fb9a650d','03001627_326a0c5116e410cfe6c1a071a4e8216c','03636649_8a6944062cbf8b25ef0a7c6e0ed55209','04256520_55f6500b52310f26352ecf815a233abb','04530566_94ddf20a9a6e035e85f7a3de54751f1b','02828884_9ad5cab6ff1e45fd48113d3612de043b','03001627_326f74afbed5d727da8b0c70313fbbae','03636649_8a9f2e5b726ea37f60ad823977adaa23','04256520_5649e603e8a9b2c295c539fc7d92aba','04530566_94e216dc57731577c14e2939682bc455','02828884_9bd9483c8eeeb703cb2a965e75be701c','03001627_3276361e9e651238ac4ed5495364a497','03691459_95d01543b46b5e43f398d0c5832df00e','04256520_5660a383172900a6593ebeeedbff73b','04530566_950ebca8ad7d94051fba2cab1ada6bf6','02933112_93ee050394f3b15e3c8d0fdfb1cc2535','03001627_3289bcc9bf8f5dab48d8ff57878739ca','03691459_96a3c3653c343db6ba8a1820ecdbe891','04256520_56652a99cd43b77744dace04559bf008','04530566_954c459bc6762abc24f2ecb72410a6d9','02933112_941289c22ad19099a87002a4eeaf610','03001627_328df096e089c4eafebad4f49b26ec52','03691459_96ff36cb731b29a61ad88f716ea80910','04256520_569c7293b52b633814038d588fd1342f','04530566_956c3b989bdd0603158a3417d0510bc','02933112_948709ebfb6cd6ff6f2222642bd41c09','03001627_329c2234d134dc89492d9da2668ec34c','03691459_97bf4aac2d956c1d5d9e9a1d5cade7db','04256520_56cafcac4df5308d35dda488a4bbb1e1','04530566_969163f17ce6467d9378da473ee38a8d','02933112_94d10abadfa0d88bf51f77a6d7299806','03001627_329ec98f10af7214ac6962daa1b6ab91','03691459_982f52d3117411a37ec3f7c14c03a92c','04256520_572da8680677fe8937b2bb75885cfc44','04530566_98cbe2b3e62879058e3175d49fbb0f30','02933112_94f4f2342f6335c5875c4d98e634f167','03001627_32a4ddf426cef33c323ad87fe7d4deee','03691459_98ad42e08a991125f0ea0ee719f6dcd1','04256520_57ceb2601802a37e534fa06200d07790','04530566_9951a6732eb8438a79662f01dd94fba1','02933112_951377627e2fb20f86d53ab0fe94e911','03001627_32a9329c11b5c35d4b3c42e318f3affc','03691459_98b920157670bcdd716d882f857922cf','04256520_57f5a7660b1f186e14038d588fd1342f','04530566_996c90952cfff5b24baa0720a34ff704','02933112_9524af09747661fbe0f91080da8431d7','03001627_32f2998a16e477163c4f66791e25960f','03691459_98cff6064749a5f3e746404352385716','04256520_580e58ca5b0f8dcf490ad276cd2af3a4','04530566_9b90b9cbd9577d842b72b4a851b36ab9','02933112_956c437be87a2be5f51f77a6d7299806','03001627_32f918efaa64a4d9c423490470c47d79','03691459_99f296d0bbae5585414ff38ecabd5968','04256520_58447a958c4af154942bb07caacf4df3','04530566_9b93b845578ee8a20c10ff7bcef26d','02933112_962b62d2b3823aff51f77a6d7299806','03001627_3358536e8e7c416ea9ef8e11754eeede','03691459_9a8a760dd2094921bb476b1cb791329b','04256520_592fdb44d4bbf0ee490ad276cd2af3a4','04530566_9c2c87ceba2a465f86b234e3f0128df2','02933112_96969d7adc540bf686d53ab0fe94e911','03001627_337050c0a5a7f965cc5cf3ad66086732','03691459_9c8dd83df9678d3bc33323f64a5f289e','04256520_59c32d74e6de63643d41bddf307a46a8','04530566_9c50b10fb651e57fdd93d77eaf89012','02933112_97b415cd78587de5fa29682ba98e856d','03001627_33aaad494817a6f4ab705559ec99536f','03691459_9ea3e05166af97ed20363e2561dd589a','04256520_5a94cc0c277f391df9aec59741c69cf7','04530566_9e3c0b7fb69ec3997cd1f8dd6fbce8fb','02933112_98d963a9f353cd026b0f9e3b3feb2454','03001627_33e436e30da86f3bc5beea20858a99d5','03691459_9ec130a3ef44b7a1e47833b310955a9f','04256520_5b23328fa6d5382d295b24579cf55b8','04530566_9e49192ba54ab0fc92c108c58096cae','02933112_99ff3359d64f1f45ce5d6e9371bb5c33','03001627_341a2d3df9800314fa260f4362cac599','03691459_9f7eb24e82020dcf40df330a2f73b17c','04256520_5b5bd4ca75b788c6ece5b3f5f7505a42','04530566_9efd4dac9e4b1698876eb99526752ffb','02933112_9a195ea2a21bc7511a4db721d603d852','03001627_348528e8d474a003cb481b0b11df1849','03691459_a05dda1538ddcb4cd747b49524a1246e','04256520_5bf5096583e15c0080741efeb2454ffb','04530566_9f468767b1fd9285eb2c303a0e0d287b']:
            continue
        out_folder = os.path.join(args.outdir, args.expname, str(args.x), obj_name)
        print(f'Rendering {dpath}')
        print(f'images will be saved to {out_folder}')
        os.makedirs(out_folder, exist_ok=True)

        # save the args and config files
        f = os.path.join(out_folder, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))

        if args.config is not None:
            f = os.path.join(out_folder, 'config.txt')
            if not os.path.isfile(f):
                shutil.copy(args.config, f)

        sample = dataset[d_idx]

        if use_source_list:
            obj_id = cat_name + '/' + obj_basename
            if pose_indices[obj_id][0] + args.x < 24:
                pose_index = pose_indices[obj_id][0] + args.x
            else:
                pose_index = pose_indices[obj_id][0]
        else:
            pose_index = pose_indices[0]

        data_input = dict(
            rgb_path=sample['rgb_path'],
            img_id=sample['img_id'],
            img_hw=sample['img_hw'],
            tgt_intrinsic=sample['src_intrinsics'][0:1],
            src_masks=sample['src_masks'][pose_index][None, None, :],
            src_rgbs=sample['src_rgbs'][pose_index][None, None, :],
            src_c2w_mats=sample['src_c2w_mats'][pose_index][None, None, :],
            src_intrinsics=sample['src_intrinsics'][pose_index][None, None, :],
            depth_range=sample['depth_range'][None, :]
        )

        input_im = sample['src_rgbs'][pose_index].cpu().numpy() * 255.
        input_im = input_im.astype(np.uint8)
        filename = os.path.join(out_folder, 'input.png')

        imageio.imwrite(filename, input_im)

        render_poses = sample['src_c2w_mats']
        view_indices = np.arange(0, len(render_poses), args.skip)
        render_poses = render_poses[view_indices]

        imgs = []
        with torch.no_grad():

            for idx, pose in tqdm.tqdm(zip(view_indices, render_poses), total=len(view_indices)):
                if idx != (pose_index + 5 - args.x) % 24: # not in [80, 96, 112, 128, 144, 160, 176]:
                    continue
                if not args.include_src and idx == pose_index:
                    continue
                filename = os.path.join(out_folder, f'{idx:06}.png')
                data_input['tgt_c2w_mat'] = pose[None, :]

                # load training rays
                ray_sampler = RaySamplerSingleImage(data_input, device, render_stride=1)
                ray_batch = ray_sampler.get_all()
                featmaps = model.encode(ray_batch['src_rgbs'])

                ret = render_single_image(ray_sampler=ray_sampler,
                                          ray_batch=ray_batch,
                                          model=model,
                                          projector=projector,
                                          chunk_size=args.chunk_size,
                                          N_samples=args.N_samples,
                                          inv_uniform=args.inv_uniform,
                                          N_importance=args.N_importance,
                                          det=True,
                                          white_bkgd=args.white_bkgd,
                                          render_stride=1,
                                          featmaps=featmaps)
                
                rgb_im = img_HWC2CHW(ret['outputs_fine']['rgb'].detach().cpu())
                # clamping RGB images
                rgb_im = torch.clamp(rgb_im, 0.0, 1.0)
                rgb_im = rgb_im.permute([1, 2, 0]).cpu().numpy()

                rgb_im = (rgb_im * 255.).astype(np.uint8)
                imageio.imwrite(filename, rgb_im)
                imgs.append(rgb_im)
                torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    gen_eval(args)
