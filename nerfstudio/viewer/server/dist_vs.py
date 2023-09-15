
import torch
from queue import Empty
from typing import Optional
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.viewer.viser.messages import CameraMessage

from nerfstudio.viewer.server.utils import get_intrinsics_matrix_and_camera_to_world_h


def serialize_cam_msg(m: CameraMessage) -> torch.Tensor:
    d = {"perspective":0, "fisheye":1, "equirectangular":2}
    t = torch.Tensor([
        m.aspect, m.render_aspect, m.fov, *m.matrix, d[m.camera_type],
        m.is_moving, m.timestamp])
    return t

def unserialize_cam_msg(t: torch.Tensor) -> CameraMessage:
    d = ("perspective", "fisheye", "equirectangular")
    m = CameraMessage(aspect=t[0], render_aspect=t[1], fov=t[2],
                      matrix=t[3:19], camera_type=d[t[19]], is_moving=t[20],
                      timestamp=t[21])
    return m


class DistributedRenderStateMachine:
    def __init__(self, viewer):
        self.viewer = viewer

    def _render_img(self):

        hw = torch.Tensor([0,0])
        hw = hw.to('cuda')
        self.viewer.dist.broadcast(hw, src=0)
        print('receive image_height width')

        image_height, image_width = hw[0], hw[1]

        camera: Optional[Cameras] = self.viewer.get_camera(image_height, image_width)
        assert camera is not None, "render called before viewer connected"

        camera_ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=self.viewer.get_model().render_aabb)

        self.viewer.get_model().eval()
        step = self.viewer.step

        outputs = self.viewer.get_model().get_outputs_for_camera_ray_bundle(camera_ray_bundle)

        self.viewer.get_model().train()


class DistributedViewerState:
    def __init__(self, *args, pipeline=None, dist=None, **kwargs):

        self.pipeline = pipeline
        self.dist = dist

        self.last_dist_viewer_step = 0

        self.render_statemachine = DistributedRenderStateMachine(self)

    def get_model(self):
        return self.pipeline.model

    def update_scene(self, step: int, num_rays_per_batch: Optional[int] = None,
                     dist = None,
                     dist_viewer_step: Optional[torch.Tensor] = None,
                     dist_cam_msg_t: Optional[torch.Tensor] = None) -> None:

        dist.broadcast(dist_viewer_step, src=0)
        print('receive dist_viewer_step')
        dist.broadcast(dist_cam_msg, src=0)
        print('receive dist_cam_msg')

        if dist_viewer_step.item() != self.last_dist_viewer_step:
            self.last_dist_viewer_step += 1
            self.camera_message = unserialize_cam_msg(dist_cam_msg)
            self.render_statemachine._render_img()


    def get_camera(self, image_height: int, image_width: int) -> Optional[Cameras]:
        """
        Return a Cameras object representing the camera for the viewer given the provided image height and width
        """
        cam_msg: Optional[CameraMessage] = self.camera_message
        if cam_msg is None:
            return None
        intrinsics_matrix, camera_to_world_h = get_intrinsics_matrix_and_camera_to_world_h(
            cam_msg, image_height=image_height, image_width=image_width
        )

        camera_to_world = camera_to_world_h[:3, :]
        camera_to_world = torch.stack(
            [
                camera_to_world[0, :],
                camera_to_world[2, :],
                camera_to_world[1, :],
            ],
            dim=0,
        )

        camera_type_msg = cam_msg.camera_type
        if camera_type_msg == "perspective":
            camera_type = CameraType.PERSPECTIVE
        elif camera_type_msg == "fisheye":
            camera_type = CameraType.FISHEYE
        elif camera_type_msg == "equirectangular":
            camera_type = CameraType.EQUIRECTANGULAR
        else:
            camera_type = CameraType.PERSPECTIVE

        camera = Cameras(
            fx=intrinsics_matrix[0, 0],
            fy=intrinsics_matrix[1, 1],
            cx=intrinsics_matrix[0, 2],
            cy=intrinsics_matrix[1, 2],
            camera_type=camera_type,
            camera_to_worlds=camera_to_world[None, ...],
            times=torch.tensor([self.control_panel.time], dtype=torch.float32),
        )
        camera = camera.to(self.get_model().device)
        return camera
