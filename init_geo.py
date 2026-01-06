import os, sys
import argparse
import torch
import numpy as np
from pathlib import Path
from time import time

ROOT = os.path.dirname(os.path.abspath(__file__))
TP = os.path.join(ROOT, "third_party")


CANDIDATES = [
    os.path.join(TP, "vggt"),
    os.path.join(TP, "map-anything"),
    os.path.join(TP, "UniCeption"),
]

for p in CANDIDATES:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from icecream import ic
ic(torch.cuda.is_available())  # Check if CUDA is available
ic(torch.cuda.device_count())

from mast3r.model import AsymmetricMASt3R
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.utils.device import to_numpy
from dust3r.utils.geometry import inv
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# Import additional prior models
try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    VGGT_AVAILABLE = True
except ImportError:
    VGGT_AVAILABLE = False
    print("Warning: VGGT model not available")

try:
    from mapanything.models.mapanything.model import MapAnything
    from mapanything.utils.image import load_images as mapanything_load_images
    MAPANYTHING_AVAILABLE = True
except ImportError:
    MapAnything = None
    mapanything_load_images = None
    MAPANYTHING_AVAILABLE = False
    print("Warning: MapAnything model not available")

from utils.sfm_utils import (save_intrinsics, save_extrinsic, save_points3D, save_time, save_images_and_masks,
                             init_filestructure, get_sorted_image_files, split_train_test, load_images, compute_co_vis_masks)
from utils.camera_utils import generate_interpolated_path




class ModelOutputAdapter:
    """
    Adapter for each model's output, convert them for instantsplatpp's input format. 
    """

    @staticmethod
    def vggt_adapter(vggt_output, img_shape):
        """
        convert vggt's output to mast3r/dust3r's output format
        """

        extrinsic, intrinsic = pose_encoding_to_extri_intri(vggt_output['pose_enc'], img_shape[-2:])
        vggt_output['extrinsic'] = extrinsic
        vggt_output['intrinsic'] = intrinsic

        B, S = vggt_output['extrinsic'].shape[:2]
        H, W = vggt_output['world_points'].shape[2:4]

        extrinsics_3x4 = extrinsic.cpu().numpy().squeeze(0) #[S, 3, 4]
        intrinsic_3x3 = intrinsic.cpu().numpy().squeeze(0) #[S, 3, 3]

        extrinsics_4x4 = np.zeros((S, 4, 4), dtype=np.float32)
        extrinsics_4x4[:, :3, :] = extrinsics_3x4  # Copy [R|t] part
        extrinsics_4x4[:, 3, 3] = 1.0 

        pts3d = vggt_output['world_points'].cpu().numpy().squeeze(0).astype(np.float32) #[S, H, W, 3]
        confs = vggt_output['world_points_conf'].cpu().numpy().squeeze(0).astype(np.float32)


        depth_5d = vggt_output["depth"].cpu().numpy().squeeze(0)  # [S, H, W, 1]
        depthmaps = depth_5d.squeeze(-1).reshape(S, -1).astype(np.float32)

        imgs = vggt_output["images"].cpu().numpy().squeeze(0)  # [S, 3, H, W]
        imgs = np.transpose(imgs, (0, 2, 3, 1)).astype(np.float32)  # [S, H, W, 3] - CHW to HWC
        imgs = imgs / 255.0 if imgs.max() > 1.0 else imgs  # Normalize to [0,1]
        
        focals = intrinsic_3x3[:, 0:1, 0].astype(np.float32)  # [S, 1]
    
        adapter_output = {
            'extrinsics_w2c': extrinsics_4x4,
            'intrinsics': intrinsic_3x3.astype(np.float32),
            'pts3d': pts3d,
            'confs': confs,
            'depthmaps': depthmaps,
            'imgs': imgs,
            'focals': focals,
        }

        return adapter_output

    @staticmethod
    def mapanything_adapter(mapanything_output, img_shape=None):
        """
        Convert MapAnything predictions (list[dict]) to EXACTLY the vggt_adapter schema.
        Output:
        extrinsics_w2c: (S, 4, 4) float32
        intrinsics:     (S, 3, 3) float32
        pts3d:          (S, H, W, 3) float32
        confs:          (S, H, W) float32
        depthmaps:      (S, H*W) float32
        imgs:           (S, H, W, 3) float32 in [0,1]
        focals:         (S, 1) float32
        """
        import torch

        def to_np(x):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu()
                try:
                    return x.numpy()
                except Exception:
                    return x.float().numpy()
            return np.asarray(x)

        assert isinstance(mapanything_output, (list, tuple)) and len(mapanything_output) > 0, \
            "MapAnything predictions must be a non-empty list"

        all_pts3d, all_confs, all_depth2d, all_imgs = [], [], [], []
        all_intrinsics, all_extrinsics, all_focals = [], [], []

        for i, pred in enumerate(mapanything_output):
            for k in ("pts3d", "conf", "depth_z", "img_no_norm", "intrinsics", "camera_poses"):
                if k not in pred:
                    raise KeyError(f"MapAnything prediction {i} missing key: '{k}'")

            pts3d = to_np(pred["pts3d"])            # (1,H,W,3) -> (H,W,3)
            conf  = to_np(pred["conf"])             # (1,H,W)   -> (H,W)
            depth = to_np(pred["depth_z"])          # (1,H,W,1) -> (H,W)
            img   = to_np(pred["img_no_norm"])      # (1,H,W,3) -> (H,W,3)
            K     = to_np(pred["intrinsics"])       # (1,3,3)   -> (3,3)
            c2w   = to_np(pred["camera_poses"])     # (1,4,4)   -> (4,4)

            # ---- squeeze leading singleton dims ----
            if pts3d.ndim == 4 and pts3d.shape[0] == 1:
                pts3d = pts3d[0]
            if conf.ndim == 3 and conf.shape[0] == 1:
                conf = conf[0]
            if depth.ndim == 4 and depth.shape[0] == 1:
                depth = depth[0]
            if img.ndim == 4 and img.shape[0] == 1:
                img = img[0]
            if K.ndim == 3 and K.shape[0] == 1:
                K = K[0]
            if c2w.ndim == 3 and c2w.shape[0] == 1:
                c2w = c2w[0]

            # depth -> (H, W)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            elif depth.ndim != 2:
                depth = np.squeeze(depth)
            assert depth.ndim == 2, f"depth must be (H,W), got {depth.shape}"

            # img -> (H,W,3) in [0,1]
            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
                img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            if img.max() > 1.0:
                img = img / 255.0
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)

            # pts3d -> (H,W,3)
            if pts3d.ndim == 2 and pts3d.shape[-1] == 3:
                H, W = depth.shape
                assert pts3d.shape[0] == H * W, f"pts3d is (N,3) with N={pts3d.shape[0]}, expected H*W={H*W}"
                pts3d = pts3d.reshape(H, W, 3)
            elif pts3d.ndim != 3:
                raise ValueError(f"pts3d must be (H,W,3), got {pts3d.shape}")

            # extrinsics: world->cam
            w2c = np.linalg.inv(c2w).astype(np.float32)

            # focals as (1,) for stacking to (S,1)
            fx = np.asarray([K[0, 0]], dtype=np.float32)

            all_pts3d.append(pts3d.astype(np.float32))
            all_confs.append(conf.astype(np.float32))
            all_depth2d.append(depth.astype(np.float32))
            all_imgs.append(img.astype(np.float32))
            all_intrinsics.append(K.astype(np.float32))
            all_extrinsics.append(w2c)
            all_focals.append(fx)

        pts3d        = np.stack(all_pts3d, axis=0)                   # (S, H, W, 3)
        confs        = np.stack(all_confs, axis=0)                   # (S, H, W)
        depthmaps_2d = np.stack(all_depth2d, axis=0)                 # (S, H, W)
        imgs         = np.stack(all_imgs, axis=0)                    # (S, H, W, 3)
        intrinsics   = np.stack(all_intrinsics, axis=0)              # (S, 3, 3)
        extrinsics   = np.stack(all_extrinsics, axis=0)              # (S, 4, 4)
        focals       = np.stack(all_focals, axis=0)                  # (S, 1)

        S, H, W = depthmaps_2d.shape[0], depthmaps_2d.shape[-2], depthmaps_2d.shape[-1]
        depthmaps = depthmaps_2d.reshape(S, -1).astype(np.float32)

        return {
            'extrinsics_w2c': extrinsics.astype(np.float32),
            'intrinsics':      intrinsics.astype(np.float32),
            'pts3d':           pts3d.astype(np.float32),
            'confs':           confs.astype(np.float32),
            'depthmaps':       depthmaps.astype(np.float32),
            'imgs':            imgs.astype(np.float32),
            'focals':          focals.astype(np.float32),
        }



def load_one_model(model_type, ckpt_path, device):
    """
    Load a single model based on the model_type

    Args:
        model_type (str): Type of model to load ('mast3r', 'vggt', 'lsm', 'spatial_tracker_v2')
        ckpt_path (str): Path to the model checkpoint
        device: Device to load the model on

    Returns:
        Loaded model instance
    """
    print(f"Loading {model_type} model on {device}...")

    if model_type.lower() == 'mast3r':
        return AsymmetricMASt3R.from_pretrained(ckpt_path).to(device)

    elif model_type.lower() == 'vggt':

        if not VGGT_AVAILABLE:
            raise ImportError("VGGT model is not available. Please install vggt.")

        model = VGGT()

        if os.path.exists(ckpt_path):
            print(f"Loading VGGT from local path {ckpt_path}...")
            try:
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
            except Exception as e:
                print(f"Warning: Could not load VGGT from local path {ckpt_path}: {e}")
                print("Using VGGT model from HuggingFace")
        else:
            try:
                print("Loading VGGT model from HuggingFace: facebook/VGGT-1B")
                _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
                model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

            except Exception as e:
                print(f"Warning: Could not load VGGT pretrained weights: {e}")
                print("Using randomly initialized VGGT model")
        
        return model.to(device).eval()

    elif model_type.lower() == 'mapanything':
        if not MAPANYTHING_AVAILABLE:
            raise ImportError("MapAnything model is not available. Please install map-anything.")
        
        # MapAnything can be loaded from HuggingFace or local checkpoint
        if os.path.exists(ckpt_path):
            print(f"Loading MapAnything from local path {ckpt_path}...")
            try:
                # Load from local checkpoint
                model = MapAnything.from_pretrained(ckpt_path).to(device)
            except Exception as e:
                print(f"Warning: Could not load MapAnything from local path {ckpt_path}: {e}")
                print("Using MapAnything model from HuggingFace")
                model = MapAnything.from_pretrained("facebook/map-anything").to(device)
        else:
            try:
                print("Loading MapAnything model from HuggingFace: facebook/map-anything")
                model = MapAnything.from_pretrained("facebook/map-anything").to(device)
                #print model architecture
                # with open('model_arch.txt', 'w') as f: print(model, file=f);
                # import sys; sys.exit(0);
            except Exception as e:
                print(f"Warning: Could not load MapAnything pretrained weights: {e}")
                raise ImportError("Failed to load MapAnything model")
        
        return model.eval()
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: 'mast3r', 'vggt', 'lsm', 'spatial_tracker_v2', 'mapanything'")

def run_model_inference(model, model_type, images, image_files, device, schedule, lr, niter, focal_avg):
    """
    run the model inference on chosen model, return the output in mast3r/dust3r's output format
    """
    if model_type == 'mast3r':
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=1, verbose=True)
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr, focal_avg=focal_avg)
        
        result = {
            'extrinsics_w2c': inv(to_numpy(scene.get_im_poses())).astype(np.float32),
            'intrinsics': to_numpy(scene.get_intrinsics()).astype(np.float32),
            'pts3d': np.array(to_numpy(scene.get_pts3d()), dtype=np.float32),
            'confs': np.array([param.detach().cpu().numpy() for param in scene.im_conf]).astype(np.float32),
            'depthmaps': to_numpy(scene.im_depthmaps.detach().cpu().numpy()).astype(np.float32),
            'imgs': np.array(scene.imgs).astype(np.float32),
            'focals': to_numpy(scene.get_focals()).astype(np.float32)
        }

    elif model_type == 'vggt':
        images_tensor = load_and_preprocess_images(image_files).to(device)

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
             with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images_tensor)
        result = ModelOutputAdapter.vggt_adapter(predictions, images_tensor.shape)

    elif model_type == 'mapanything':
        # Load images using MapAnything's image loader
        views = mapanything_load_images(image_files)
        
        # Run MapAnything inference
        predictions = model.infer(
            views,
            memory_efficient_inference=False,  # Set to True for large scenes
            use_amp=True,                     # Use mixed precision
            amp_dtype="bf16",                 # Use bf16 if available
            apply_mask=True,                  # Apply masking to outputs
            mask_edges=True,                  # Remove edge artifacts
            apply_confidence_mask=False,      # Don't filter by confidence
            confidence_percentile=10,         # Bottom percentile to remove
        )

        def _describe_mapanything(preds, max_items=3):
            import torch, numpy as np
            def _shape(x):
                if isinstance(x, torch.Tensor): return f"torch{tuple(x.shape)} device={x.device} dtype={x.dtype}"
                if isinstance(x, np.ndarray):   return f"np{tuple(x.shape)} dtype={x.dtype}"
                return f"{type(x)}"
            print(f"[MAPANYTHING] type={type(preds)} len={len(preds) if isinstance(preds, (list,tuple)) else 'N/A'}")
            for i, p in enumerate(preds[:max_items]):
                print(f"  item #{i}: keys={list(p.keys())}")
                for k in ("pts3d","conf","depth_z","img_no_norm","intrinsics","camera_poses"):
                    if k in p:
                        print(f"    {k:12s}: {_shape(p[k])}")
                    else:
                        print(f"    {k:12s}: MISSING")
            print("---- end describe ----")

        _describe_mapanything(predictions)

        
        # Convert MapAnything output to expected format
        result = ModelOutputAdapter.mapanything_adapter(predictions, None)
    
    return result


# def load_prior_model(model_type, ckpt_path, device):
#     """
#     Factory function to load different prior models based on model_type.
    
#     Args:
#         model_type (str): Type of model to load ('mast3r', 'vggt', 'lsm', 'spatial_tracker_v2')
#         ckpt_path (str): Path to the model checkpoint
#         device: Device to load the model on
        
#     Returns:
#         Loaded model instance
#     """
#     if model_type.lower() == 'mast3r':
#         return AsymmetricMASt3R.from_pretrained(ckpt_path).to(device)
    
#     elif model_type.lower() == 'vggt':
#         if VGGTModel is None:
#             raise ImportError("VGGT model is not available. Please install the required dependencies.")
#         return VGGTModel.from_pretrained(ckpt_path).to(device)
    
#     elif model_type.lower() == 'lsm':
#         if LSMModel is None:
#             raise ImportError("LSM model is not available. Please install the required dependencies.")
#         return LSMModel.from_pretrained(ckpt_path).to(device)
    
#     elif model_type.lower() == 'spatial_tracker_v2':
#         if SpatialTrackerV2Model is None:
#             raise ImportError("SpatialTracker v2 model is not available. Please install the required dependencies.")
#         return SpatialTrackerV2Model.from_pretrained(ckpt_path).to(device)
    
#     else:
#         raise ValueError(f"Unknown model type: {model_type}. Supported types: 'mast3r', 'vggt', 'lsm', 'spatial_tracker_v2'")


def main(source_path, model_path, ckpt_path, device, batch_size, image_size, schedule, lr, niter, 
         min_conf_thr, llffhold, n_views, co_vis_dsp, depth_thre, conf_aware_ranking=False, focal_avg=False, infer_video=False, model_type='mast3r'):

    # ---------------- (1) Load model and images ----------------  
    save_path, sparse_0_path, sparse_1_path = init_filestructure(Path(source_path), n_views)
    model = load_one_model(model_type, ckpt_path, device)

    image_dir = Path(source_path) / 'images'
    image_files, image_suffix = get_sorted_image_files(image_dir)
    if infer_video:
        train_img_files = image_files
    else:
        train_img_files, test_img_files = split_train_test(image_files, llffhold, n_views, verbose=True)
    
    # when geometry init, only use train images
    image_files = train_img_files
    images, org_imgs_shape = load_images(image_files, size=image_size)

    start_time = time()
    #old pipeline
    # print(f'>> Making pairs...')
    # pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    # print(f'>> Inference...')
    # output = inference(pairs, model, device, batch_size=1, verbose=True)
    # print(f'>> Global alignment...')
    # scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer)
    # loss = scene.compute_global_alignment(init="mst", niter=300, schedule=schedule, lr=lr, focal_avg=args.focal_avg)

    result = run_model_inference(model, model_type, images, image_files, device, schedule, lr, niter, focal_avg)

    # Extract scene information
    #old
    # extrinsics_w2c = inv(to_numpy(scene.get_im_poses()))
    # intrinsics = to_numpy(scene.get_intrinsics())
    # focals = to_numpy(scene.get_focals())
    # imgs = np.array(scene.imgs)
    # pts3d = to_numpy(scene.get_pts3d())
    # pts3d = np.array(pts3d)
    # depthmaps = to_numpy(scene.im_depthmaps.detach().cpu().numpy())
    # values = [param.detach().cpu().numpy() for param in scene.im_conf]
    # confs = np.array(values)
    extrinsics_w2c = result['extrinsics_w2c']
    intrinsics = result['intrinsics']
    focals = result['focals']
    imgs = result['imgs']
    pts3d = result['pts3d']
    depthmaps = result['depthmaps']
    confs = result['confs']

    if conf_aware_ranking:
        print(f'>> Confiden-aware Ranking...')
        avg_conf_scores = confs.mean(axis=(1, 2))
        sorted_conf_indices = np.argsort(avg_conf_scores)[::-1]
        sorted_conf_avg_conf_scores = avg_conf_scores[sorted_conf_indices]
        print("Sorted indices:", sorted_conf_indices)
        print("Sorted average confidence scores:", sorted_conf_avg_conf_scores)
    else:
        sorted_conf_indices = np.arange(n_views)
        print("Sorted indices:", sorted_conf_indices)

    # Calculate the co-visibility mask
    print(f'>> Calculate the co-visibility mask...')
    if depth_thre > 0:
        overlapping_masks = compute_co_vis_masks(sorted_conf_indices, depthmaps, pts3d, intrinsics, extrinsics_w2c, imgs.shape, depth_threshold=depth_thre)
        overlapping_masks = ~overlapping_masks
    else:
        co_vis_dsp = False
        overlapping_masks = None
    end_time = time()
    Train_Time = end_time - start_time
    print(f"Time taken for {n_views} views: {Train_Time} seconds")
    save_time(model_path, '[1] coarse_init_TrainTime', Train_Time)

    # ---------------- (2) Interpolate training pose to get initial testing pose ----------------
    if not infer_video:
        n_train = len(train_img_files)
        n_test = len(test_img_files)

        if n_train < n_test:
            n_interp = (n_test // (n_train-1)) + 1
            all_inter_pose = []
            for i in range(n_train-1):
                tmp_inter_pose = generate_interpolated_path(poses=extrinsics_w2c[i:i+2], n_interp=n_interp)
                all_inter_pose.append(tmp_inter_pose)
            all_inter_pose = np.concatenate(all_inter_pose, axis=0)
            all_inter_pose = np.concatenate([all_inter_pose, extrinsics_w2c[-1][:3, :].reshape(1, 3, 4)], axis=0)
            indices = np.linspace(0, all_inter_pose.shape[0] - 1, n_test, dtype=int)
            sampled_poses = all_inter_pose[indices]
            sampled_poses = np.array(sampled_poses).reshape(-1, 3, 4)
            assert sampled_poses.shape[0] == n_test
            inter_pose_list = []
            for p in sampled_poses:
                tmp_view = np.eye(4)
                tmp_view[:3, :3] = p[:3, :3]
                tmp_view[:3, 3] = p[:3, 3]
                inter_pose_list.append(tmp_view)
            pose_test_init = np.stack(inter_pose_list, 0)
        else:
            indices = np.linspace(0, extrinsics_w2c.shape[0] - 1, n_test, dtype=int)
            pose_test_init = extrinsics_w2c[indices]

        save_extrinsic(sparse_1_path, pose_test_init, test_img_files, image_suffix)
        test_focals = np.repeat(focals[0], n_test)
        save_intrinsics(sparse_1_path, test_focals, org_imgs_shape, imgs.shape, save_focals=False)
    # -----------------------------------------------------------------------------------------

    # Save results
    focals = np.repeat(focals[0], n_views)
    print(f'>> Saving results...')
    end_time = time()
    save_time(model_path, '[1] init_geo', end_time - start_time)
    save_extrinsic(sparse_0_path, extrinsics_w2c, image_files, image_suffix)
    save_intrinsics(sparse_0_path, focals, org_imgs_shape, imgs.shape, save_focals=True)
    pts_num = save_points3D(sparse_0_path, imgs, pts3d, confs.reshape(pts3d.shape[0], -1), overlapping_masks, use_masks=co_vis_dsp, save_all_pts=True, save_txt_path=model_path, depth_threshold=depth_thre)
    save_images_and_masks(sparse_0_path, n_views, imgs, overlapping_masks, image_files, image_suffix)
    print(f'[INFO] {model_type} Reconstruction is successfully converted to COLMAP files in: {str(sparse_0_path)}')
    print(f'[INFO] Number of points: {pts3d.reshape(-1, 3).shape[0]}')    
    print(f'[INFO] Number of points after downsampling: {pts_num}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and save results.')
    parser.add_argument('--source_path', '-s', type=str, required=True, help='Directory containing images')
    parser.add_argument('--model_path', '-m', type=str, required=True, help='Directory to save the results')
    parser.add_argument('--model_type', type=str, default='mast3r', 
        choices=['mast3r', 'vggt', 'lsm', 'spatial_tracker_v2', 'mapanything'],
        help='Type of prior model to use')
    
    # Set default checkpoint paths based on model type
    default_checkpoints = {
        'mast3r': './mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth',
        'vggt': './vggt/checkpoints/VGGT_model.pth',
        'lsm': './lsm/checkpoints/LSM_model.pth',
        'spatial_tracker_v2': './spatial_tracker_v2/checkpoints/SpatialTrackerV2_model.pth',
        'mapanything': 'facebook/map-anything'  # HuggingFace model ID
    }
    
    parser.add_argument('--ckpt_path', type=str, default=default_checkpoints['mast3r'], help='Path to the model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing images')
    parser.add_argument('--image_size', type=int, default=512, help='Size to resize images')
    parser.add_argument('--schedule', type=str, default='cosine', help='Learning rate schedule')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--niter', type=int, default=300, help='Number of iterations')
    parser.add_argument('--min_conf_thr', type=float, default=5, help='Minimum confidence threshold')
    parser.add_argument('--llffhold', type=int, default=8, help='')
    parser.add_argument('--n_views', type=int, default=3, help='')
    # parser.add_argument('--focal_avg', type=bool, default=False, help='')
    parser.add_argument('--focal_avg', action="store_true")
    parser.add_argument('--conf_aware_ranking', action="store_true")
    parser.add_argument('--co_vis_dsp', action="store_true")
    parser.add_argument('--depth_thre', type=float, default=0.01, help='Depth threshold')
    parser.add_argument('--infer_video', action="store_true")
    

    args = parser.parse_args()
    
    # Set default checkpoint path based on model type if not explicitly provided
    if args.ckpt_path == default_checkpoints['mast3r'] and args.model_type != 'mast3r':
        args.ckpt_path = default_checkpoints[args.model_type]
    
    main(args.source_path, args.model_path, args.ckpt_path, args.device, args.batch_size, args.image_size, args.schedule, args.lr, args.niter,         
          args.min_conf_thr, args.llffhold, args.n_views, args.co_vis_dsp, args.depth_thre, args.conf_aware_ranking, args.focal_avg, args.infer_video, args.model_type)