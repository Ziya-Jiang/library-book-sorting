"""V3.4: direct visual servo; ordinary robot actions retain CuRobo planning."""
import argparse
from contextlib import nullcontext
import sys
from pathlib import Path
from cart_book_grasp_v3_column import load_config
CONFIG_PATH=Path(__file__).with_name('cart_book_grasp_v3_4_column.json')
from cart_book_grasp_v3_4_policy import load_stream_config
CONFIG=Path(__file__).with_name('cart_book_grasp_v3_4_stream.json')
from cart_book_grasp_v3_4_wrist_gpt import wrist_fallback
from cart_book_grasp_v3_camera import WristCamera


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--column-config',type=Path,default=CONFIG_PATH)
    parser.add_argument('--stream-config',type=Path,default=CONFIG)
    parser.add_argument('--check-config',action='store_true')
    parser.add_argument('--motion-config',type=Path,
                        default=Path(__file__).with_name('cart_book_grasp_v3_config.json'),
                        help='Hot-loaded grasp distances; legacy V3 default is unchanged')
    selection_group=parser.add_mutually_exclusive_group()
    selection_group.add_argument('--select-book-number', action='store_true',
                        help='Show fresh head book numbers and wait for one selection each cycle')
    selection_group.add_argument('--batch-right-to-left', action='store_true',
                        help='Observe head once, then grasp and place each book from robot right to left')
    selection_group.add_argument('--plug', action='store_true', help='Two-pass horizontal book insertion')
    parser.add_argument('--selector-config', type=Path,
                        help='Head catalog config for numbered or batch selection')
    args,remaining=parser.parse_known_args(argv)
    catalog_mode=args.select_book_number or args.batch_right_to_left or args.plug
    if args.selector_config and not catalog_mode:
        parser.error('--selector-config requires --select-book-number, --batch-right-to-left or --plug')
    if catalog_mode:
        from cart_book_grasp_v3_4_numbered import CONFIG as NUMBERED_CONFIG, check_config, head_selection
        selector_path = args.selector_config or NUMBERED_CONFIG
        check_config(selector_path, batch=args.batch_right_to_left or args.plug, plug=args.plug)
        print('V3.4 批次模式：头部只识别一次，从右到左逐本抓放；腕部实时伺服保持原后端。'
              if args.batch_right_to_left else '插槽模式：抬Z→退到初始X→头部实时找槽→等高横移/插入→降Z；先右移后左移。' if args.plug else
              'V3.4 编号模式：只抓本轮选中编号，禁用自动换书；腕部视觉伺服保持原后端。',flush=True)
    c=load_config(args.column_config);s=load_stream_config(args.stream_config)
    print('V3.4 头部分割：使用共享源最高原始分辨率 RGB-D，不使用缩略图。',flush=True)
    from cart_book_grasp_v3_2_head_fallback import load_config as load_head_fallback
    from cart_book_grasp_v2_gpt import provider_label
    head_fallback=load_head_fallback()
    model_label=f"{provider_label(head_fallback)} / {head_fallback['model']}"
    print(f'V3.4 腕部备用：SAM3 拒绝后直接使用 {model_label} 圈定区域；不使用 SAM2 或边界算法。',flush=True)
    if args.batch_right_to_left:
        print(f'V3.4 批次目标：首次{model_label}整书/书脊及实测深度固定；不输入编号、不复拍比较，完成一遍结束。',flush=True)
    elif args.select_book_number:
        print(f'V3.4 编号目标：{model_label}直接圈定多本书及对应书脊→实测左右编号→输入编号；不调用SAM3编号，不自动换书。',flush=True)
    elif args.plug:
        print('插槽目标：每轮头部重新识别书本；每本抬升抽出后头部RGB-D实时找槽。保持抬升高度插入独立配置距离，最终X处竖直放下，确认后松爪。',flush=True)
    else:
        print(f"V3.4 头部三维书脊倾角超过 {head_fallback.get('tilt_review_threshold_deg',45.):g}°：GPT 复核同一本书，再以实测深度重算。",flush=True)
    if c['mode']!='centerline_distance':parser.error('V3.4 requires centerline_distance')
    # V3.4 translation/gain are independently bounded by load_stream_config.
    # The shared column max_step_mm remains the original blocking V3 setting.
    if s['lookahead_roll_deg']>c['max_roll_step_deg']:
        parser.error('Streaming roll lookahead exceeds existing rotation limit')
    camera=WristCamera()
    try:
        if c['target_u'] is not None and not 2<=c['target_u']<camera.size[0]-2:parser.error('target_u outside image')
        print(f"V3.4 滚动目标：gain={s['gain']:g}，最多 {s['update_hz']:g} Hz 更新；中间不等到位，探测和终点仍等待。",flush=True)
        print(f"V3.4 前伸目标距离上限 {s['forward_lead_mm']:g} mm；偏差接近恢复阈值时收短，恢复后 {s['forward_ramp_seconds']:g}s 渐增。",flush=True)
        print('视觉阶段使用 CPU Jacobian 关节伺服；复位、粗定位和抓取仍使用 CuRobo。',flush=True)
        if args.check_config:
            if remaining:parser.error('Unknown check arguments: '+' '.join(remaining))
            print('配置和标定通过；未连接设备。运行时复用 V3.1 直接伺服适配器（10176）；原规划器不变。');return
        planner_scripts=Path(__file__).resolve().parent.parent/'dual_v2_2_control/cuarm_upper_planner/scripts'
        sys.path.insert(0,str(planner_scripts))
        from v3_1_readiness import require_ready
        require_ready()
        from cart_book_grasp_v3_1_ports import dedicated_planner
        import dual_v2_2_agent_pcl as agent
        from cart_book_grasp_v1_3 import main as v13_main
        from cart_book_grasp_v3_motion import installed
        from cart_book_grasp_v3_4_reselection import enabled as ik_reselection
        root=Path(__file__).parent;saved=sys.argv;sys.argv=[saved[0],*remaining]
        saved_camera_factory=agent.start_calibrated_orbbec
        import head_rgbd.client as head_client
        agent.start_calibrated_orbbec=lambda calibration,warmup_frames=5: head_client.CalibratedCamera(
            calibration,warmup_frames=warmup_frames,prefer_native=True)
        try:
            selection = head_selection(selector_path,batch=args.batch_right_to_left or args.plug,plug=args.plug) if catalog_mode else nullcontext(v13_main)
            reselection = nullcontext() if catalog_mode else ik_reselection()
            with selection as run_entry, reselection, dedicated_planner(agent), installed(None,column_path=args.column_config,stream_config=s,wrist_boundary_selector=wrist_fallback):
                run_entry(config_path=args.motion_config,wrist_config_path=root/'cart_book_grasp_v3_wrist_config.json')
        finally:
            sys.argv=saved
            agent.start_calibrated_orbbec=saved_camera_factory
    finally:camera.close()


if __name__=='__main__':
    try:main()
    except (ValueError,OSError) as error:
        print(f'V3.4: {error}',file=sys.stderr);raise SystemExit(2)
