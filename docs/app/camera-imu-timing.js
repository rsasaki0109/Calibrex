"use strict";
(() => {
  const vector=v=>Array.isArray(v)&&v.length===3&&v.every(Number.isFinite);
  function validate(data){
    if(!data||data.schema!=="calibrex.camera_imu_timing_demo/v1"||data.capture_kind!=="real"||!Number.isFinite(data.saved_offset_s)||!Array.isArray(data.shift_range_ms)||data.shift_range_ms[0]!==-50||data.shift_range_ms[1]!==50)throw Error("Invalid measured clock data");
    const t=data.imu_time_s,c=data.gyro_cumulative_camera_deg,a=data.camera_start_s,b=data.camera_end_s,r=data.camera_rate_dps;
    if(!Array.isArray(t)||t.length<2||t.length>20000||!t.every((v,i)=>Number.isFinite(v)&&(i===0||v>t[i-1]))||!Array.isArray(c)||c.length!==t.length||!c.every(vector)||!Array.isArray(a)||!a.length||a.length>1000||!Array.isArray(b)||b.length!==a.length||!Array.isArray(r)||r.length!==a.length||!r.every(vector))throw Error("Invalid measured clock series");
    if(!a.every((v,i)=>Number.isFinite(v)&&Number.isFinite(b[i])&&b[i]>v&&(i===0||v>a[i-1])&&v+data.saved_offset_s-.05>=t[0]&&b[i]+data.saved_offset_s+.05<=t.at(-1)))throw Error("Clock range requires extrapolation");
    return data;
  }
  function cumulativeAt(t,times,values){
    if(t<times[0]||t>times.at(-1))throw Error("No IMU support at this time");
    let lo=0,hi=times.length-1;
    while(hi-lo>1){const m=(lo+hi)>>1;if(times[m]<=t)lo=m;else hi=m;}
    const f=(t-times[lo])/(times[hi]-times[lo]);
    return values[lo].map((v,k)=>v+f*(values[hi][k]-v));
  }
  function evaluate(data,shiftMs){
    if(!Number.isFinite(shiftMs)||shiftMs<-50||shiftMs>50)throw Error("Clock shift must be within ±50 ms");
    const offset=data.saved_offset_s+shiftMs/1000,axisSquares=[0,0,0],predicted=[],residuals=[];
    data.camera_start_s.forEach((start,i)=>{
      const end=data.camera_end_s[i],a=cumulativeAt(start+offset,data.imu_time_s,data.gyro_cumulative_camera_deg),b=cumulativeAt(end+offset,data.imu_time_s,data.gyro_cumulative_camera_deg);
      const p=a.map((v,k)=>(b[k]-v)/(end-start)),error=p.map((v,k)=>v-data.camera_rate_dps[i][k]);
      error.forEach((v,k)=>axisSquares[k]+=v*v);predicted.push(p);residuals.push(error);
    });
    return {offset_s:offset,predicted,residuals,rmse_dps:Math.sqrt(axisSquares.reduce((s,v)=>s+v,0)/predicted.length),axis_rmse_dps:axisSquares.map(v=>Math.sqrt(v/predicted.length))};
  }
  const COLOURS=["#5ce4d0","#ffba74"];
  class CameraImuTiming {
    constructor(panel,json){
      this.panel=panel;this.json=json;this.sequence=0;this.shift=0;this.axis=0;this.point=0;
      panel.innerHTML=`<div class="method-media"><div class="cloud-topline"><span>REAL HILTI / CLOCK OFFSET EXPERIMENT</span><span>Fixed saved rotation and gyro bias</span></div><div class="timing-controls" hidden><div class="timing-toolbar"><label>Axis <select aria-label="Clock comparison axis"><option value="0">Camera X</option><option value="1">Camera Y</option><option value="2">Camera Z</option></select></label><strong class="timing-offset"></strong></div><label class="timing-shift-label">Injected shift relative to saved estimate <input type="range" min="-50" max="50" step="1" value="0" aria-label="Injected Camera-IMU clock shift"></label><div class="cloud-controls"><button type="button" data-shift="-50">−50 ms</button><button type="button" data-shift="-10">−10 ms</button><button type="button" data-shift="0">Reset · 0 ms</button><button type="button" data-shift="10">+10 ms</button><button type="button" data-shift="50">+50 ms</button></div><p class="timing-caption">Camera and gyro curves are means over the same camera intervals · deg/s</p><canvas class="timing-rates" width="960" height="280" role="img" aria-label="Measured camera and interval-averaged gyro angular rates"></canvas><div class="timing-legend"><span>Camera-derived rate</span><span>Gyro mean in camera frame</span></div><p class="timing-caption">Selected-axis residual · gyro minus camera · deg/s</p><canvas class="timing-residuals" width="960" height="160" role="img" aria-label="Selected-axis angular-rate residual"></canvas><p class="timing-caption">Displayed-window 3D residual RMSE versus injected shift · click to inspect</p><canvas class="timing-profile" width="960" height="170" role="img" aria-label="Angular-rate residual sensitivity to clock shift"></canvas><div class="timing-inspect"><label>Inspect interval <input type="range" min="0" max="0" value="0" aria-label="Clock comparison camera interval"></label><output></output></div></div><p class="method-status" role="status">Loading measured clock series…</p><button class="timing-retry" type="button" hidden>Retry clock data</button></div><aside class="cloud-evidence"><div class="eyebrow">CAMERA ↔ IMU / CHANGE ONLY THE CLOCK</div><h2>A few milliseconds.<br>A different match.</h2><p class="cloud-description">Move the IMU sampling window against the measured camera intervals. Rotation and gyro bias stay at the saved Calibrex estimate.</p><div class="timing-metrics"></div><p class="cloud-description timing-convention"></p><p class="cloud-description">Zero injection is the saved estimate, not clock truth. This development-window residual combines camera tracking noise and model errors. No optimizer or new calibration verdict is running.</p><p class="cloud-source">Hilti 2022 · exp21 / cam0<br>Hilti SLAM Challenge 2022 terms</p><a class="method-link" href="../assets/camera_imu_lock/timing/">Data, clock convention &amp; limits ↗</a><a class="method-link" href="../assets/camera-imu-lock.gif">Earlier recorded alignment replay ↗</a></aside>`;
      panel.querySelector("select").addEventListener("change",e=>{this.axis=Number(e.target.value);this.draw();});
      panel.querySelector(".timing-shift-label input").addEventListener("input",e=>this.setShift(Number(e.target.value)));
      panel.querySelectorAll("[data-shift]").forEach(b=>b.addEventListener("click",()=>this.setShift(Number(b.dataset.shift))));
      panel.querySelector(".timing-inspect input").addEventListener("input",e=>{this.point=Number(e.target.value);this.draw();});
      panel.querySelector(".timing-retry").addEventListener("click",()=>this.load());
      panel.querySelector(".timing-profile").addEventListener("click",e=>{if(!this.data)return;const box=e.currentTarget.getBoundingClientRect(),x=(e.clientX-box.left)/box.width*960;this.setShift(Math.max(-50,Math.min(50,Math.round(-50+(x-58)/872*100))));});
    }
    async load(){
      const seq=++this.sequence,panel=this.panel;this.data=null;panel.querySelector(".timing-controls").hidden=true;panel.querySelector(".timing-retry").hidden=true;
      panel.querySelector(".method-status").textContent="Loading measured clock series…";
      panel.querySelector(".timing-metrics").replaceChildren();
      try{
        const data=validate(await this.json("samples/camera_imu_timing.json"));if(seq!==this.sequence)return;
        this.data=data;this.profile=Array.from({length:101},(_,i)=>evaluate(data,i-50));
        this.point=Math.min(this.point,data.camera_start_s.length-1);panel.querySelector(".timing-inspect input").max=String(data.camera_start_s.length-1);
        this.rateBounds=[0,1,2].map(k=>{const values=[...data.camera_rate_dps.map(p=>p[k]),...this.profile.flatMap(r=>r.predicted.map(p=>p[k]))];return [Math.min(...values),Math.max(...values)];});
        this.residualBounds=[0,1,2].map(k=>{const bound=Math.max(...this.profile.flatMap(r=>r.residuals.map(p=>Math.abs(p[k]))));return [-bound,bound];});
        panel.querySelector(".timing-controls").hidden=false;this.setShift(this.shift);
        panel.querySelector(".timing-convention").textContent=`t_IMU = t_camera + ${(data.saved_offset_s*1000).toFixed(3)} ms + injected shift. A positive shift samples later IMU times. All ${data.camera_start_s.length} intervals remain the same; no extrapolation.`;
        panel.querySelector(".method-status").textContent="Measured Hilti series loaded · interval means recomputed locally · no optimizer running";
      }catch(error){if(seq===this.sequence){panel.querySelector(".method-status").textContent=error.message;panel.querySelector(".timing-retry").hidden=false;}}
    }
    setShift(value){this.shift=value;this.panel.querySelector(".timing-shift-label input").value=String(value);this.draw();}
    draw(){
      if(!this.data)return;
      const d=this.data,k=this.axis,result=this.profile[this.shift+50],baseline=this.profile[50],axis=["X","Y","Z"][k],n=d.camera_start_s.length;
      this.panel.querySelector(".timing-offset").textContent=`${this.shift>=0?"+":""}${this.shift} ms injected`;
      const x=d.camera_start_s.map((v,i)=>(v+d.camera_end_s[i])/2),camera=d.camera_rate_dps.map(v=>v[k]),gyro=result.predicted.map(v=>v[k]),residual=result.residuals.map(v=>v[k]);
      this.plot(".timing-rates",x,[camera,gyro],this.rateBounds[k],this.point,"s","deg/s");
      this.plot(".timing-residuals",x,[residual],this.residualBounds[k],this.point,"s","deg/s");
      const profileValues=this.profile.map(v=>v.rmse_dps);
      this.plot(".timing-profile",this.profile.map((_,i)=>i-50),[profileValues],[Math.min(...profileValues),Math.max(...profileValues)],this.shift+50,"ms","deg/s");
      const rows=[["Displayed-window 3D RMSE",result.rmse_dps.toFixed(3)+" deg/s"],["Saved estimate · 0 ms injection",baseline.rmse_dps.toFixed(3)+" deg/s"],[`Selected ${axis}-axis RMSE`,result.axis_rmse_dps[k].toFixed(3)+" deg/s"],["Total candidate time offset",(result.offset_s*1000).toFixed(3)+" ms"],["Same camera intervals",String(n)]];
      const metrics=this.panel.querySelector(".timing-metrics");metrics.replaceChildren();
      for(const [label,value] of rows){const card=document.createElement("div"),small=document.createElement("small"),strong=document.createElement("strong");small.textContent=label;strong.textContent=value;card.append(small,strong);metrics.append(card);}
      this.panel.querySelector(".timing-inspect input").value=String(this.point);
      this.panel.querySelector(".timing-inspect output").textContent=`Interval ${this.point+1}/${n}: ${d.camera_start_s[this.point].toFixed(3)}–${d.camera_end_s[this.point].toFixed(3)} s · ${axis}: camera ${camera[this.point].toFixed(2)}, gyro ${gyro[this.point].toFixed(2)}, residual ${residual[this.point].toFixed(2)} deg/s`;
      this.panel.querySelector(".timing-rates").setAttribute("aria-label",`Camera ${axis} interval-average rates, injected offset ${this.shift} milliseconds, displayed-window 3D RMSE ${result.rmse_dps.toFixed(3)} degrees per second.`);
    }
    plot(selector,x,series,bounds,cursor,xunit,yunit){
      const canvas=this.panel.querySelector(selector),ctx=canvas.getContext("2d"),w=canvas.width,h=canvas.height;
      const [lo,hi]=bounds,span=hi-lo||1,xmin=x[0],xmax=x.at(-1),px=v=>58+(v-xmin)/(xmax-xmin||1)*(w-88),py=v=>h-34-(v-lo+span*.1)/(span*1.2)*(h-62);
      ctx.fillStyle="#091524";ctx.fillRect(0,0,w,h);ctx.strokeStyle="#24394e";ctx.lineWidth=1;
      for(let i=0;i<=4;i++){const yy=28+i*(h-62)/4;ctx.beginPath();ctx.moveTo(58,yy);ctx.lineTo(w-30,yy);ctx.stroke();}
      if(lo<=0&&hi>=0){ctx.strokeStyle="#8eabc177";ctx.beginPath();ctx.moveTo(58,py(0));ctx.lineTo(w-30,py(0));ctx.stroke();}
      series.forEach((values,k)=>{ctx.strokeStyle=COLOURS[k];ctx.lineWidth=2;ctx.beginPath();values.forEach((v,i)=>i?ctx.lineTo(px(x[i]),py(v)):ctx.moveTo(px(x[i]),py(v)));ctx.stroke();});
      ctx.strokeStyle="#fff8";ctx.beginPath();ctx.moveTo(px(x[cursor]),18);ctx.lineTo(px(x[cursor]),h-30);ctx.stroke();
      ctx.fillStyle="#a7bfd4";ctx.font="12px monospace";ctx.fillText(hi.toFixed(1),5,27);ctx.fillText(lo.toFixed(1),5,h-29);ctx.fillText(`${xmin.toFixed(1)} ${xunit}`,58,h-8);ctx.fillText(`${xmax.toFixed(1)} ${xunit}`,w-110,h-8);ctx.fillText(yunit,w-80,18);
    }
  }
  globalThis.CalibrexTiming={validate,cumulativeAt,evaluate,CameraImuTiming};
})();
