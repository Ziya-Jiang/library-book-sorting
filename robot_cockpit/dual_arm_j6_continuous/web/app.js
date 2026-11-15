import {Viewer} from './viewer.js';
const $=id=>document.getElementById(id);let selected=1,snapshot=null,socket=null,viewer=null,inputs=[],connected=false;
let token=new URLSearchParams(location.hash.slice(1)).get('token')||sessionStorage.getItem('dual-arm-control-token');if(token){sessionStorage.setItem('dual-arm-control-token',token);history.replaceState(null,'',location.pathname);}
const error=text=>$('error').textContent=text;
function send(action,data={}){if(!connected||!snapshot||socket?.readyState!==WebSocket.OPEN){error('浏览器尚未连接');return;}socket.send(JSON.stringify({action,challenge:snapshot.challenge,epoch:snapshot.epoch,...data}));if(action!=='heartbeat')error('');}
function rows(){const parent=$('joints');parent.replaceChildren();inputs=[];for(let j=0;j<7;j++){const row=document.createElement('div');row.className='joint';const label=document.createElement('div'),title=document.createElement('strong'),meta=document.createElement('small');title.textContent=`J${j+1}   —`;meta.textContent='等待反馈';label.append(title,meta);const input=document.createElement('input');input.type='number';input.step='0.1';input.placeholder='目标 °';input.setAttribute('aria-label',`J${j+1} 目标角度`);const button=document.createElement('button');button.textContent='执行';button.disabled=true;button.onclick=()=>{if(input.value.trim()===''){error('请输入目标角度');return;}send('joint',{arm:selected,joint:j,degrees:Number(input.value),duration:Number($('duration').value)});};row.append(label,input,button);parent.append(row);inputs.push({title,meta,input,button});}}
function render(s){snapshot=s;const p=s.planner||{};$('plannerStatus').textContent=p.status||'未连接 Planner';$('plannerEnable').disabled=!(s.fresh&&!s.active&&p.connected);$('plannerDisable').disabled=!p.enabled;const manual=s.mine&&s.fresh&&!p.enabled; $('connect').disabled=!connected; $('plannerConfig').disabled=!connected;$('releaseControl').disabled=!(s.active&&s.mine&&!p.enabled);$('initialPose').disabled=!(manual&&s.initial_pose_degrees);$('zeroPose').disabled=!manual;for (const [arm, buttons] of [['leftHandOpen', 'leftHandClose'], ['rightGripperOpen', 'rightGripperClose']].entries()) {
    for (const id of buttons) $(id).disabled = !(manual && s.capabilities?.gripper && s.gripper_installed?.[arm]);
  }$('rtlog').textContent=s.rt_log||'暂无日志';$('mode').textContent=s.simulation?'仿真模式':'真实硬件';$('connection').textContent=s.fresh?'RT 在线':'无实时反馈';$('connection').className=s.fresh?'live':'stale';$('status').textContent=s.message;$('claim').disabled=!s.fresh||s.active;$('claim').textContent=manual?'已接管关节控制':'接管关节控制';$('start').disabled=!connected||s.rt_running||$('location').value!=='local';$('stoprt').disabled=!connected||!s.rt_running;$('telemetry').textContent=s.fresh?`${s.simulation?'仿真':'机器人'}反馈 · ${s.age_ms} ms · ${['启动','空闲','运动','等待','故障','恢复','关闭'][s.state.system]||'未知'} · 诊断 0x${s.state.diagnostics.toString(16)}`:'无实时反馈 · 模型仅为参考/最后姿态';$('startupHint').textContent=s.simulation?'软件仿真；不会打开硬件。接管后可验证关节与三维模型。':'真实硬件：启动 RT 会初始化旧驱动，并可能使能电机/释放制动器。接受当前姿态只更新启动参考，不校准机械零位。';
 inputs.forEach(({title,meta,input,button},j)=>{button.disabled=!manual;if(s.state){const a=s.state.arms[selected],deg=a.position[j]*180/Math.PI;title.textContent=`J${j+1}   ${deg.toFixed(2)}°`;meta.textContent=`${a.rt_position?'RT '+(a.rt_position[j]*180/Math.PI).toFixed(1)+'° · ':''}速度 ${(a.velocity[j]*180/Math.PI).toFixed(2)}°/s · ${a.enabled[j]?'已使能':'未使能'}`;input.min=(a.soft_position[j][0]*180/Math.PI).toFixed(2);input.max=(a.soft_position[j][1]*180/Math.PI).toFixed(2);if(input.value===''&&document.activeElement!==input)input.value=deg.toFixed(1);}else{title.textContent=`J${j+1}   —`;meta.textContent='反馈不可用';}});$('leftTool').textContent = s.gripper_installed?.[0] ? '左臂 · 夹爪' : ($('showReservedTool').checked ? '左夹爪 · 预留未安装' : '左臂 · 空法兰');
  $('rightTool').textContent = s.gripper_installed?.[1] ? '右臂 · 夹爪' : '右臂 · 空法兰';
  viewer?.update(s.state, s.gripper_installed);
 if(s.mine&&document.visibilityState==='visible')send('heartbeat');
}
$('plannerConfig').onclick=()=>send('planner_config',{ip:$('plannerIp').value,port:Number($('plannerPort').value),local_ip:$('plannerLocalIp').value,local_port:Number($('plannerLocalPort').value)});
$('plannerEnable').onclick=()=>send('planner_enable'); $('plannerDisable').onclick=()=>send('stop');
$('leftHandOpen').onclick=()=>send('gripper',{gripper:0,gesture:'open'});$('leftHandClose').onclick=()=>send('gripper',{gripper:0,gesture:'close'});$('rightGripperOpen').onclick=()=>send('gripper',{gripper:1,gesture:'open'});$('rightGripperClose').onclick=()=>send('gripper',{gripper:1,gesture:'close'});
$('initialPose').onclick=()=>{if(!snapshot?.mine||!snapshot?.fresh||snapshot?.planner?.enabled||!snapshot?.initial_pose_degrees){error('请先接管关节控制');return;}send('initial_pose',{duration:Number($('duration').value)});};$('zeroPose').onclick=()=>{if(!snapshot?.mine||!snapshot?.fresh||snapshot?.planner?.enabled){error('请先接管关节控制');return;}send('zero_pose',{duration:Number($('duration').value)});};$('releaseControl').onclick=()=>send('stop');
$('stop').onclick=()=>send('stop');$('claim').onclick=()=>send('claim');$('start').onclick=()=>send('start_rt',{reference:$('reference').value});$('stoprt').onclick=()=>send('stop_rt');$('connect').onclick=()=>send('connect',{ip:$('location').value==='local'?'127.0.0.1':$('ip').value,port:Number($('port').value)});$('location').onchange=()=>{$('ip').disabled=$('location').value==='local';$('start').disabled=$('location').value!=='local'||snapshot?.rt_running;};$('ip').disabled=true;$('home').onclick=()=>viewer?.reset();document.querySelectorAll('[data-arm]').forEach(b=>b.onclick=()=>{selected=Number(b.dataset.arm);document.querySelectorAll('[data-arm]').forEach(x=>x.classList.toggle('selected',x===b));rows();if(snapshot)render(snapshot);});
window.addEventListener('keydown',e=>{if(e.key==='Escape'){e.preventDefault();send('stop');}});document.addEventListener('visibilitychange',()=>{if(document.hidden&&snapshot?.mine&&!snapshot?.planner?.enabled)send('stop');});window.addEventListener('pagehide',()=>{if(snapshot?.mine&&!snapshot?.planner?.enabled)send('stop');});
$('showReservedTool').onchange = () => {
  if (viewer) viewer.showReservedTool = $('showReservedTool').checked;
  if (snapshot) render(snapshot);
};
rows();
function connect(){if(!token){error('请使用 runtime/web.json 中带访问口令的完整链接');return;}socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws`);socket.onopen=()=>{connected=true;socket.send(JSON.stringify({token}));};socket.onmessage=e=>{const m=JSON.parse(e.data);if(m.type==='state')render(m);if(m.type==='error')error(m.message);};socket.onclose=()=>{connected=false;if(snapshot)render({...snapshot,active:false,mine:false,fresh:false,state:null,message:'浏览器已断连；控制权已释放'});setTimeout(connect,1000);};socket.onerror=()=>error('连接失败');}
connect();

try {
  viewer = new Viewer($('view'));
  viewer.showReservedTool = $('showReservedTool').checked;
  const response = await fetch('/model.json', {cache:'no-store'});
  if (!response.ok) throw Error('模型清单读取失败');
  const model = await response.json();
  await viewer.load(model);
  if (snapshot) viewer.update(snapshot.state, snapshot.gripper_installed);
  $('source').textContent = '已安装双夹爪 URDF · 实测关节 · 夹爪指令回显';
} catch (e) {
  $('source').textContent = '三维模型加载失败：' + e.message;
}
