from pathlib import Path

def paths_npy2jpg(path_to_npy: str):
    """
    Convert the npy files to png files
    example input: '/datasets/Carrada/2019-09-16-12-55-51/range_angle_processed/000002.npy'
    example output: '/datasets/Carrada/2019-09-16-12-55-51/camera_images/000002.jpg'

    Parameters
    ----------
    path_to_npy : str
        Path to the npy file to convert
    """
    img_name = path_to_npy.split('/')[-1].split('.')[0] + '.jpg'
    return {
        'img_num': path_to_npy.split('/')[-1].split('.')[0],
        'img_seq': path_to_npy.split('/')[-3],
        'img_path': Path(path_to_npy).parent.parent / 'camera_images' / img_name,
    }
