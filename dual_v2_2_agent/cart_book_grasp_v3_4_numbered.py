"""Numbered head selection composed with the existing V3.4 servo hooks.

Selection runs on the PC using the configured head source. The normal V3.4
wrist/board runtime remains responsible for the visual approach on 155.
"""
from contextlib import contextmanager, nullcontext
from pathlib import Path

CONFIG = Path(__file__).with_name('cart_book_grasp_v2_1_config.json')


def check_config(path=CONFIG, *, batch=False, plug=False):
    if plug:
        from book_plug import PlugSelectionAgent as Selector
    elif batch:
        from cart_book_grasp_v3_4_batch import BatchBookSelectionAgent as Selector
    else:
        from cart_book_grasp_v3_4_gpt_numbered import GPTNumberedBookSelectionAgent as Selector
    selector = Selector(path)
    selector.check_ready()
    return selector


@contextmanager
def head_selection(path=CONFIG, *, batch=False, plug=False):
    import cart_book_grasp_v1_3_motion as motion
    import cart_book_grasp_v1_3_visualization as display
    import cart_book_grasp_v2_motion as shell_motion
    from cart_book_grasp_v2_1_motion import install_selected_spine
    from cart_book_grasp_v3_4_numbered_review import review_locked_spine
    from cart_book_grasp_v2_1_visualization import install_book_visualization
    from cart_book_grasp_v2_1_numbered import NumberedCartLoop
    from cart_book_grasp_v2_dual_view import load_config
    from cart_book_grasp_v2 import main as numbered_main
    from cart_book_grasp_v1_3_perception import main as perception_main

    selector = check_config(path, batch=batch, plug=plug)
    import head_rgbd.client as head_client
    saved_camera = head_client.CalibratedCamera

    def camera_factory(*args, **kwargs):
        camera = saved_camera(*args, **kwargs)
        if batch:
            return selector.attach_camera(camera)
        selector.observation_camera = camera
        return camera
    # The optional V2 wrist-context snapshot uses local USB. Disable only that
    # snapshot; V3.4 selects/tracks with its existing remote wrist camera later.
    dual = dict(load_config(), enabled=False)

    def install(geometry, visualization=None, on_geometry=None, **unused):
        if plug:
            selector.visualization = visualization
        # GPT auto-reselection is intentionally not used for a numbered target.
        return install_selected_spine(geometry, visualization, on_geometry,
                                      selector=selector, dual_config=dual,
                                      selected_review=review_locked_spine)
    install.numbered_selection = True

    if plug:
        from book_plug import PlugCartLoop as BaseLoop
    elif batch:
        from cart_book_grasp_v3_4_batch import BatchCartLoop
        BaseLoop = BatchCartLoop
    else:
        BaseLoop = NumberedCartLoop

    class BoardNumberedLoop(BaseLoop):
        startup_message = (BaseLoop.startup_message if batch else
                           'V3.4 编号选书：GPT-6识别并显示编号→输入编号→Y对准→腕部视觉伺服抓取；'
                           '每轮重新编号，失败不自动换书。')

        def next_task(self):
            # agent.main invokes this only after camera, models and Rerun setup.
            # No first_reset, numbering or task command may precede the grant.
            from book_startup_gate import wait_for_launcher
            wait_for_launcher(lambda: self.stop.is_set() or self.pause.is_set())
            return super().next_task()

    def run(*, config_path, wrist_config_path):
        import cart_book_grasp_v2_visualization as numbered_display
        previous = shell_motion._CART_PLUS_CONFIG_PATH
        previous_preview = numbered_display.preview_targets
        previous_reset_reload = shell_motion._reload_initial_right_j6
        shell_motion._CART_PLUS_CONFIG_PATH = Path(config_path)
        if plug or Path(config_path).resolve() == Path(__file__).with_name('book_grasp_motion.json').resolve():
            # Only the one-command entry uses the new motion policy. Keep the
            # historical V2 visualization and other entrypoints unchanged.
            numbered_display.preview_targets = display.preview_targets
            shell_motion._reload_initial_right_j6 = motion._reload_initial_right_j6
        try:
            if plug:
                from book_plug import installed as plug_runtime
            with plug_runtime(selector) if plug else nullcontext():
                return numbered_main(selector=selector, loop_factory=BoardNumberedLoop,
                                     book_entry=perception_main, config_path=config_path,
                                     wrist_config_path=wrist_config_path)
        finally:
            shell_motion._CART_PLUS_CONFIG_PATH = previous
            numbered_display.preview_targets = previous_preview
            shell_motion._reload_initial_right_j6 = previous_reset_reload

    saved_install = motion._install_spine_center
    saved_display = display.install_book_visualization
    motion._install_spine_center = install
    display.install_book_visualization = install_book_visualization
    head_client.CalibratedCamera = camera_factory
    try:
        yield run
    finally:
        motion._install_spine_center = saved_install
        display.install_book_visualization = saved_display
        head_client.CalibratedCamera = saved_camera
