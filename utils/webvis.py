from pathlib import Path
import http.server
import socketserver
import numpy as np
import scenepic as sp


def default_camera():
    r = 2.0
    theta = np.pi / 12
    gamma = np.pi / 4
    return sp.Camera(
        center=np.array([
            -r * np.cos(theta) * np.cos(gamma),
            -r * np.cos(theta) * np.sin(gamma),
             r * np.sin(theta)
        ]),
        up_dir=np.array([0., 0., 1.])
    )

def start_http_server(path, PORT):
    DIRECTORY = str(path)
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=DIRECTORY, **kwargs)
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        httpd.serve_forever()


def to_scenepic_mesh(scene, mesh, color, name):
    mesh_sp = scene.create_mesh(name)
    mesh_sp.shared_color = color
    mesh_sp.add_mesh_without_normals(
        vertices=mesh.vertices,
        triangles=mesh.faces
    )
    return mesh_sp


def vis_meshes(meshes, filename):
    scene = sp.Scene()

    # scene.grid(width="600px", grid_template_rows="400px 400px 400px", grid_template_columns="600px")

    colors = [
        np.array([0.7, 0.7, 0.7]),
        np.array([0.0, 0.0, 1.0])
    ]

    meshes_sp = []

    for k, mesh in enumerate(meshes):
        # mesh = mesh_data["mesh"]
        # case = mesh_data["case"]
        mesh_sp = to_scenepic_mesh(scene, mesh, colors[k], f"mesh_{k}")
        meshes_sp.append(mesh_sp)

    main = scene.create_canvas_3d(width=900, height=600)
    main.shading = sp.Shading(
        bg_color=sp.Colors.White
    )
    main.camera = default_camera()
    frame1 = main.create_frame(meshes=meshes_sp)
    # scene.place(main.canvas_id, str(j+1), "1")

    scene.save_as_html(filename, title="Meshes")


def vis_pointcloud(scene, pts, rgb):
    mesh = scene.create_mesh("mesh")
    mesh.shared_color = np.array([0.7, 0.7, 0.7])
    
    mesh.add_cube()
    mesh.apply_transform(sp.Transforms.Scale(0.002)) 
    mesh.enable_instancing(positions=pts,
                           colors=rgb)
    return mesh


def ground_plane_mesh(scene, z_val):
    mesh_sp = scene.create_mesh("ground_plane")
    mesh_sp.shared_color = np.array([0.0, 0.3, 0.7])
    vertices = np.array([
        [-1, -1, z_val],
        [-1, 1, z_val],
        [1, 1, z_val],
        [1, -1, z_val]
    ], dtype=np.float64)
    faces = np.array([
        [0, 2, 1],
        [0, 3, 2]
    ], dtype=np.int64)
    mesh_sp.double_sided = True
    mesh_sp.add_mesh_without_normals(
        vertices=vertices,
        triangles=faces
    )
    return mesh_sp


def bbox_mesh(scene, bx):
    mesh_sp = scene.create_mesh("ground_plane")
    mesh_sp.shared_color = np.array([0.0, 0.3, 0.7])
    x,y,z = bx
    mesh_sp.add_quad(
        p0=[-x, -y, -z],
        p1=[-x, y, -z],
        p2=[x, y, -z],
        p3=[x, -y, -z],
        add_wireframe=True,
        fill_triangles=False
    )
    mesh_sp.add_quad(
        p0=[-x, -y, z],
        p1=[-x, y, z],
        p2=[x, y, z],
        p3=[x, -y, z],
        add_wireframe=True,
        fill_triangles=False
    )
    mesh_sp.add_quad(
        p0=[-x, -y, -z],
        p1=[-x, y, -z],
        p2=[-x, y, z],
        p3=[-x, -y, z],
        add_wireframe=True,
        fill_triangles=False
    )
    mesh_sp.add_quad(
        p0=[x, -y, -z],
        p1=[x, y, -z],
        p2=[x, y, z],
        p3=[x, -y, z],
        add_wireframe=True,
        fill_triangles=False
    )
    return mesh_sp


def vis_mesh(filename,
             mesh,
             half_bbox_size=None,
             vis_axes=False):
    scene = sp.Scene()
    mesh_sp = scene.create_mesh("mesh")
    mesh_sp.shared_color = np.array([0.7, 0.7, 0.7])
    mesh_sp.add_mesh_without_normals(
        vertices=mesh.vertices,
        triangles=mesh.faces
    )
    main = scene.create_canvas_3d(width=900, height=600)
    main.shading = sp.Shading(
        bg_color=sp.Colors.White
    )
    main.camera = default_camera()

    all_meshes = [mesh_sp]

    if vis_axes:
        axes = scene.create_mesh(f"axes")
        axes.add_coordinate_axes()
        all_meshes += [axes]

    if half_bbox_size is not None:
        all_meshes += [bbox_mesh(scene, half_bbox_size)]

    frame1 = main.create_frame(meshes=all_meshes)

    # scene.link_canvas_events(main)
    scene.save_as_html(filename, title="Meshes")
