import numpy as np
import pymeshlab
from pathlib import Path
from rich.progress import track

# Find the neighboring face sharing two vertices with the current face
def find_neighbor(faces, vertex_face_map, v1, v2, exclude_face_idx):
    for idx in (vertex_face_map[v1] & vertex_face_map[v2]):
        if idx != exclude_face_idx:
            face = faces[idx].tolist()
            face.remove(v1)
            face.remove(v2)
            return idx
    return exclude_face_idx

if __name__ == '__main__':
    # Input/output paths
    root = Path('XXXXXXXXXXXXXXXX')
    out_root = Path('XXXXXXXXXXXXXXXX')

    # List all .obj mesh files
    shape_list = sorted(root.glob('*.obj'))

    ms = pymeshlab.MeshSet()

    for shape_path in track(shape_list, description="Processing meshes..."):
        out_path = out_root / shape_path.relative_to(root).with_suffix('.npz')
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ms.clear()
        ms.load_new_mesh(str(shape_path))
        mesh = ms.current_mesh()

        vertices = mesh.vertex_matrix()
        faces = mesh.face_matrix()

        # Normalize: center and scale mesh
        center = (np.max(vertices, axis=0) + np.min(vertices, axis=0)) / 2
        vertices -= center
        max_len = np.max(np.sum(vertices**2, axis=1))
        vertices /= np.sqrt(max_len)

        # Recalculate normals after normalization
        ms.clear()
        ms.add_mesh(pymeshlab.Mesh(vertices, faces))
        face_normals = ms.current_mesh().face_normal_matrix()

        # Build vertex-to-face lookup table
        vertex_face_map = [set() for _ in range(len(vertices))]
        centers, corners = [], []

        for i, (v1, v2, v3) in enumerate(faces):
            # Add face index to each involved vertex
            for v in (v1, v2, v3):
                vertex_face_map[v].add(i)

            # Compute face center and corner coordinates
            p1, p2, p3 = vertices[v1], vertices[v2], vertices[v3]
            centers.append((p1 + p2 + p3) / 3)
            corners.append(np.concatenate([p1, p2, p3]))

        # Identify neighbor faces
        neighbors = [
            [
                find_neighbor(faces, vertex_face_map, v1, v2, i),
                find_neighbor(faces, vertex_face_map, v2, v3, i),
                find_neighbor(faces, vertex_face_map, v3, v1, i)
            ]
            for i, (v1, v2, v3) in enumerate(faces)
        ]

        # Combine features
        centers = np.array(centers)
        corners = np.array(corners)
        all_features = np.concatenate([centers, corners, face_normals], axis=1)
        neighbors = np.array(neighbors)

        # Save to compressed .npz format
        np.savez(out_path, face=all_features, neighbor_index=neighbors)