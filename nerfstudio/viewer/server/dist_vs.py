
from queue import Empty
from typing import Optional

class DistributedRenderStateMachine:
    def __init__(self, viewer):
        self.viewer = viewer

    def _render_img(self, camera_ray_bundle):

        self.viewer.get_model().eval()
        step = self.viewer.step

        outputs = self.viewer.get_model().get_outputs_for_camera_ray_bundle(camera_ray_bundle)

        self.viewer.get_model().train()


class DistributedViewerState:
    def __init__(self, *args, pipeline=None, queue=None, **kwargs):

        self.pipeline = pipeline
        self.queue = queue

        self.render_statemachine = DistributedRenderStateMachine(self)

    def get_model(self):
        return self.pipeline.model

    def update_scene(self, step: int, num_rays_per_batch: Optional[int] = None):
        try:
            action = queue.get_nowait()
            print('Get action')
        except Empty:
            return

        if action[0] == 'Render':
            self.render_statemachine._render_img(action[1])

