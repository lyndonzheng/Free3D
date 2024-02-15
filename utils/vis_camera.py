import argparse
import scenepic as sp
import numpy as np
from omegaconf import DictConfig
from skimage.transform import downscale_local_mean

from utils.webvis import default_camera, start_http_server


"""
Code borrowed from:
https://github.com/Kai-46/nerfplusplus/blob/master/camera_visualizer/visualize_cameras.py
"""
def get_camera_frustum(img_size, K, W2C, frustum_length=0.5, color=[0., 1., 0.]):
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],                          # frustum origin
                               [-half_w, -half_h, frustum_length],    # top-left image corner
                               [half_w, -half_h, frustum_length],     # top-right image corner
                               [half_w, half_h, frustum_length],      # bottom-right image corner
                               [-half_w, half_h, frustum_length]])    # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    C2W = np.linalg.inv(W2C)
    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]
    return frustum_points, frustum_lines, frustum_colors


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    return merged_points, merged_lines, merged_colors

def vis_pointcloud(scene, pts, rgb):
    """visualize the point cloud with the color"""
    mesh = scene.create_mesh("mesh")
    mesh.shared_color = np.array([0.7, 0.7, 0.7])
    
    mesh.add_cube()
    mesh.apply_transform(sp.Transforms.Scale(0.002)) 
    mesh.enable_instancing(positions=pts,
                           colors=rgb)
    return mesh


def vis_cameras(scene, dataset, frustum_size=0.5, nth=1):
    "visualize all the camera pose"
    frustums = []
    for k in range(0, len(dataset), nth):
        # read image
        img_data = dataset[k].image_rgb.permute(1, 2, 0).numpy()
        # read camera
        cam = dataset[k].camera
        h, w = img_data.shape[:2]
        K = np.zeros((3,3))
        K[0,0], K[1,1] = float(cam.focal_length[0][0]) * min(h,w) / 2, float(cam.focal_length[0][1]) * min(h,w) / 2
        K[0,2], K[1,2], K[2, 2] = (w-float(cam.principal_point[0][0])*min(h,w))/2, (h-float(cam.principal_point[0][1])*min(h,w))/2, 1.0
        W2C = cam.get_world_to_view_transform().get_matrix()[0].transpose(-1,-2).numpy()
        W2C[[0,1], :] *= -1.0
        frustum = get_camera_frustum((h,w), K, W2C, frustum_length=frustum_size)
        frustums.append(frustum)

    points, lines, colors = frustums2lineset(frustums)
    lines = lines.astype(np.int64)

    mesh = scene.create_mesh(shared_color=sp.Color(0.0, 1.0, 0.0))
    mesh.add_lines(
        start_points=points[lines[:, 0], :],
        end_points=points[lines[:, 1], :]
    )

    return mesh

def vis_images(scene, dataset, frustum_size=0.5, nth=1):
    """visualize all the projection rgb images"""
    
    meshes = []
    for k in range(0, len(dataset), nth):
        texture_id = f"frame_{k:02}"
        img = scene.create_image(image_id=texture_id)
        # read image
        img_data = dataset[k].image_rgb.permute(1, 2, 0).numpy()
        img_data_scaled = downscale_local_mean(img_data, (2, 2, 1))
        img.from_numpy(img_data_scaled)
        # read camera
        import pdb
        pdb.set_trace()
        cam = dataset[k].camera
        h, w = img_data.shape[:2]
        K = np.zeros((3,3))
        K[0,0], K[1,1] = float(cam.focal_length[0][0]) * min(h,w) / 2, float(cam.focal_length[0][1]) * min(h,w) / 2
        K[0,2], K[1,2], K[2, 2] = (w-float(cam.principal_point[0][0])*min(h,w))/2, (h-float(cam.principal_point[0][1])*min(h,w))/2, 1.0
        W2C = cam.get_world_to_view_transform().get_matrix()[0].transpose(-1,-2).numpy()
        W2C[[0,1], :] *= -1.0
        stuff = get_camera_frustum((h,w), K, W2C, frustum_length=frustum_size)
        frustum_points = stuff[0]
        frustum_image_points = frustum_points[1:, :]
        # build the mesh
        mesh = scene.create_mesh(texture_id=texture_id)
        mesh.double_sided = True
        mesh.add_mesh_without_normals(
            frustum_image_points,
            np.array([[0, 2, 1], [0, 3, 2]], dtype=np.uint32),
            uvs=np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
        )
        meshes.append(mesh)

    return meshes


def main(args):
    expand_args_fields(JsonIndexDatasetMapProviderV2)
    dataset = JsonIndexDatasetMapProviderV2(
        category = args.category,
        subset_name = args.subset_name,
        dataset_root = args.root_dir,
        test_on_train=False,
        only_test_set=False,
        load_eval_batches=False,
        # n_known_frames_for_test=args.n_views,
        dataset_JsonIndexDataset_args=DictConfig({
                "remove_empty_masks": True,
                "sort_frames": True,
                "load_point_clouds": True,
                "pick_sequence": args.instance,
                "image_height": 800,
                "image_width": 576,
                }),
    ).get_dataset_map()

    train_dataset = dataset['train']
    # load the pointcoloud
    pts = train_dataset[0].sequence_point_cloud.points_padded().squeeze().numpy()
    rgb = train_dataset[0].sequence_point_cloud.features_padded().squeeze().numpy()

    RED = [1., 0., 0.]
    GREEN = [0., 1., 0.]
    BLUE = [0., 0., 1.]
    import pdb
    pdb.set_trace()
    # load all cameras
    # cameras = train_dataset.get_all_train_cameras()
    # pick example
    # cameras = [cameras[0], cameras[10], cameras[20]]
    train_dataset = [train_dataset[0], train_dataset[40], train_dataset[80]]
    # represent the scene
    scene = sp.Scene()
    # visualize point cloud
    point_cloud = vis_pointcloud(scene, pts, rgb)
    # visualize images
    image_meshes = vis_images(scene, train_dataset, frustum_size=args.frustum_size, nth=args.nth)
    # visualize cameras
    all_cameras = [vis_cameras(scene, train_dataset, frustum_size=args.frustum_size, nth=args.nth)]
    all_meshes = [point_cloud] + image_meshes + all_cameras
    # output the visualized results to html
    main = scene.create_canvas_3d(width=1600, height=1600,
                                  shading=sp.Shading(bg_color=sp.Colors.White))
    main.camera = default_camera()
    # visualize the results
    frame1 = main.create_frame(meshes=all_meshes)
    filename = 'index.html'
    scene.save_as_html(filename, title=args.category+'_'+args.instance)
    start_http_server(".", 8097)


def vis_points_images_cameras(W2C, K, imgs, pts=None, rgb=None, frustum_size=0.5, nth=1, filename='index.html'):
    # represent the scene
    scene = sp.Scene()
    # visualize cameras and imgs
    frustums = []
    img_meshes = []
    b, h, w, c = imgs.size()
    for i in range(0, b, nth):
        # read camera
        frustum = get_camera_frustum((h, w), K[i], W2C[i], frustum_length=frustum_size)
        # read image
        texture_id = f"frame_{i:02}"
        img = scene.create_image(image_id=texture_id)
        # read image
        img_data = (imgs[i].numpy() + 1) / 2.0
        img_data_scaled = downscale_local_mean(img_data, (2,2,1))
        img.from_numpy(img_data_scaled)
        frustum_points = frustum[0]
        frustum_image_points = frustum_points[1:, :]
        # build the mesh
        img_mesh = scene.create_mesh(texture_id=texture_id)
        img_mesh.double_sided = True
        img_mesh.add_mesh_without_normals(
            frustum_image_points,
            np.array([[0, 2, 1], [0, 3, 2]], dtype=np.uint32),
            uvs=np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
        )
        img_meshes.append(img_mesh)
        frustums.append(frustum)
    points, lines, colors = frustums2lineset(frustums)
    lines = lines.astype(np.int64)
    cam_meshes = scene.create_mesh(shared_color=sp.Color(0.0, 1.0, 0.0))
    cam_meshes.add_lines(
        start_points=points[lines[:, 0], :],
        end_points=points[lines[:, 1], :]
    )
    if pts is not None and rgb is not None:
        point_mesh = scene.create_mesh("mesh")
        point_mesh.shared_color = np.array([0.7, 0.7, 0.7])
        point_mesh.add_cube()
        point_mesh.apply_transform(sp.Transforms.Scale(0.002))
        point_mesh.enable_instancing(positions=pts, colors=rgb)
        all_meshes = [point_mesh] + img_meshes + [cam_meshes]
    else:
        all_meshes = img_meshes + [cam_meshes]
    # output the visualized results to html
    main = scene.create_canvas_3d(width=1600, height=1600,
                                  shading=sp.Shading(bg_color=sp.Colors.White))
    main.camera = default_camera()
    # visualize the results
    frame1 = main.create_frame(meshes=all_meshes)
    scene.save_as_html(filename, title=filename.split('.')[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualise training camera frustums and images.')
    parser.add_argument(
        '--root_dir', 
        type=str,
        default="/scratch/shared/beegfs/shared-datasets/co3dv2/many_view_subset",
        help=("data path of CO3D")
    )
    parser.add_argument(
        '--instance', 
        type=str,
        default="34_1479_4753",
        help=("Instance in CO3D dataset")
    )
    parser.add_argument(
        '--category', 
        type=str,
        default="teddybear",
        help=("CO3D category")
    )
    parser.add_argument(
        '--subset_name', 
        type=str,
        default="manyview_dev_0",
        help=("subset_name in CO3D")
    )
    parser.add_argument(
        '--nth', 
        type=int,
        default=4,
        help=("Number of rames to skip.")
    )
    parser.add_argument(
        '--frustum_size',
        type=float,
        default=4,
        help="Default frustum size",
    )
    parser.add_argument(
        '--structured-split',
        action='store_true',
        help=("Visualise cameras in the structured split only.")
    )
    parser.add_argument(
        '--x-is-reflection-axis',
        action='store_true'
    )
    parser.add_argument('--vis-axes', action='store_true')
    args = parser.parse_args()
    main(args)