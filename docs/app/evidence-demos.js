"use strict";
(() => {
  const CODES=["comparison","handeye","gnss","vehicle","radar","live"];
  const COLOURS=["#5ce4d0","#ffba74","#b9a0ff","#79b8ff"];
  const finitePoint=p=>Array.isArray(p)&&p.length>=3&&p.length<=4&&p.every(Number.isFinite);
  function validate(data,code){
    if(!data||data.schema!=="calibrex.measured_demo/v1"||data.capture_kind!=="real"||data.id!==code||!Array.isArray(data.views)||!data.views.length||data.views.length>20)throw Error("Invalid measured demo");
    for(const view of data.views){
      if(!["cloud","lines","radar"].includes(view.kind)||typeof view.label!=="string"||!Array.isArray(view.metrics)||!view.metrics.every(m=>typeof m.label==="string"&&(typeof m.value==="string"||Number.isFinite(m.value))))throw Error("Invalid measured metrics");
      if(view.kind==="lines"){
        const c=view.chart;
        if(!c||!Array.isArray(c.x)||!c.x.length||c.x.length>10000||!c.x.every(Number.isFinite)||!Array.isArray(c.series)||!c.series.length||c.series.length>8||!c.series.every(s=>Array.isArray(s.values)&&s.values.length===c.x.length&&s.values.every(Number.isFinite)))throw Error("Invalid measured chart");
      }else if(!Array.isArray(view.clouds)||!view.clouds.length||view.clouds.length>8||!view.clouds.every(c=>Array.isArray(c.points)&&c.points.length>0&&c.points.length<=10000&&c.points.every(finitePoint)))throw Error("Invalid measured coordinates");
    }
    return data;
  }
  function textNode(tag,text,className){const e=document.createElement(tag);e.textContent=text;if(className)e.className=className;return e;}
  class EvidenceDemo {
    constructor(panel,json,code){
      this.panel=panel;this.json=json;this.code=code;this.sequence=0;this.zoom=1;this.angle=-.4;this.pitch=.35;this.point=0;this.epoch=0;
      panel.innerHTML=`<div class="method-media"><div class="cloud-topline"><span class="evidence-mode">MEASURED DATA / SAVED NATIVE RESULTS</span><span class="evidence-dataset"></span></div><div class="evidence-toolbar"><label>View <select aria-label="${code} demo view"></select></label><label class="evidence-zoom">Zoom <input type="range" min="1" max="12" step=".5" value="1" aria-label="${code} plot zoom"></label></div><h3 class="evidence-plot-title"></h3><canvas class="evidence-plot" width="960" height="440" role="img" aria-label="Measured data visualization"></canvas><div class="evidence-stress" hidden><label>Observation condition <select aria-label="Live observation condition"><option value="normal">Full measured coverage</option><option value="limited">Limited coverage · 2 captures</option><option value="outliers">Inject 4 outliers · fit only</option></select></label><label>Loss <select aria-label="Live fitting loss"><option value="least_squares">Least squares</option><option value="huber">Huber · 0.05 m threshold</option></select></label><p class="evidence-scenario-note"></p><p class="evidence-quality" role="status"></p></div><div class="evidence-live-actions" hidden><button type="button" data-live="run">Run live fit</button><button type="button" data-live="step">One iteration</button><button type="button" data-live="stop">Stop</button><button type="button" data-live="reset">Reset perturbation</button></div><canvas class="evidence-history" width="960" height="180" role="img" aria-label="Actual fit and holdout residual history" hidden></canvas><div class="evidence-controls"><label class="evidence-point-label">Inspect observation <input type="range" min="0" max="0" value="0" aria-label="${code} observation"></label><output class="evidence-point-detail"></output></div><div class="evidence-extra"></div><div class="evidence-legend"></div><p class="method-status" role="status">Loading measured data…</p><button class="evidence-retry" type="button" hidden>Retry data</button></div><aside class="cloud-evidence"><div class="eyebrow">REAL OBSERVATIONS / EXPLORE THE EVIDENCE</div><h2></h2><div class="evidence-metrics"></div><p class="cloud-description evidence-note"></p><p class="cloud-source"></p><div class="evidence-targetless"></div><a class="method-link" href="../assets/evidence-demos/">Sources, methods &amp; limits ↗</a><a class="method-link" href="samples/evidence_demo_index.json">Byte-bound data receipt ↗</a></aside>`;
      this.canvas=panel.querySelector(".evidence-plot");this.context=this.canvas.getContext("2d");
      panel.querySelector("select").addEventListener("change",()=>this.select());
      panel.querySelector(".evidence-zoom input").addEventListener("input",e=>{this.zoom=Number(e.target.value);this.draw();});
      panel.querySelector(".evidence-point-label input").addEventListener("input",e=>{this.point=Number(e.target.value);this.draw();});
      panel.querySelector(".evidence-retry").addEventListener("click",()=>this.load());
      let drag;
      this.canvas.addEventListener("pointerdown",e=>{drag=[e.clientX,e.clientY];this.canvas.setPointerCapture(e.pointerId);});
      this.canvas.addEventListener("pointermove",e=>{if(!drag||this.view?.kind!=="cloud")return;this.angle+=(e.clientX-drag[0])*.008;this.pitch=Math.max(-1.4,Math.min(1.4,this.pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];this.draw();});
      for(const event of ["pointerup","pointercancel","lostpointercapture"])this.canvas.addEventListener(event,()=>drag=null);
      panel.querySelectorAll(".evidence-stress select").forEach(select=>select.addEventListener("change",()=>{this.stop();this.resetLive();this.draw();}));
      panel.querySelectorAll("[data-live]").forEach(button=>button.addEventListener("click",()=>{
        this.stop();const action=button.dataset.live;
        if(action==="reset"){this.resetLive();this.draw();}
        else if(action==="stop")panel.querySelector(".method-status").textContent=`Paused · ${this.live.iteration} actual iterations`;
        else if(action==="step")this.step();
        else if(action==="run"){const epoch=this.epoch;const tick=()=>{if(epoch!==this.epoch)return;if(this.step())this.timer=setTimeout(tick,60);};tick();}
      }));
    }
    stop(){this.epoch++;clearTimeout(this.timer);}
    async load(){
      this.stop();const seq=++this.sequence,panel=this.panel,code=this.code==="live"?"comparison":this.code;
      panel.querySelector(".method-status").textContent="Loading measured data…";panel.querySelector(".evidence-retry").hidden=true;
      this.context.clearRect(0,0,this.canvas.width,this.canvas.height);this.view=null;
      try{
        if(!CODES.includes(this.code))throw Error("Unknown demo");
        const data=validate(await this.json(`samples/evidence_${code}.json`),code);if(seq!==this.sequence)return;
        this.data=data;const select=panel.querySelector("select");select.replaceChildren();
        for(const view of this.code==="live"?[{id:"live",label:"6-DoF point least squares"}]:data.views){const o=textNode("option",view.label);o.value=view.id;select.append(o);}
        panel.querySelector("h2").textContent=this.code==="live"?"Actual iterations. Real observations.":data.title;
        panel.querySelector(".evidence-dataset").textContent=data.dataset;
        panel.querySelector(".cloud-source").textContent=data.dataset+"\n"+data.license;
        panel.querySelector(".evidence-mode").textContent=this.code==="live"?"LIVE BROWSER OPTIMIZER / REAL ACFR OBSERVATIONS":"MEASURED DATA / EXPLORE RESULTS";
        this.select();
      }catch(error){if(seq===this.sequence){panel.querySelector(".method-status").textContent=error.message;panel.querySelector(".evidence-retry").hidden=false;}}
    }
    select(){
      this.stop();this.point=0;this.zoom=1;this.panel.querySelector(".evidence-zoom input").value="1";
      this.view=this.data.views.find(v=>v.id===this.panel.querySelector("select").value)||this.data.views[0];
      if(this.code==="live")this.resetLive();
      this.panel.querySelector(".evidence-live-actions").hidden=this.code!=="live";
      this.panel.querySelector(".evidence-stress").hidden=this.code!=="live";
      this.panel.querySelector(".evidence-history").hidden=this.code!=="live";
      this.panel.querySelector(".evidence-plot-title").textContent=this.code==="live"?"Measured board centres · actual current estimate":this.view.title;
      this.panel.querySelector(".evidence-note").textContent=this.code==="live"?"A small browser Gauss–Newton point fit runs on measured ACFR centres. Only fit observations enter the updates; held-out residuals are diagnostic. The start is a deliberately perturbed saved native Horn estimate. This live example is not the full production calibration pipeline.":this.view.note;
      const extra=this.panel.querySelector(".evidence-extra");extra.replaceChildren();this.yaw=0;this.lever=[0,0];
      if(this.code==="radar")this.addControl(extra,"Radar yaw offset",-15,15,.5,v=>{this.yaw=v*Math.PI/180;this.draw();});
      if(this.code==="gnss")for(let i=0;i<2;i++)this.addControl(extra,"Candidate lever "+["forward","right"][i],-2,2,.05,v=>{this.lever[i]=v;this.draw();});
      const evidence=this.panel.querySelector(".evidence-targetless");evidence.replaceChildren();
      if(this.code==="comparison"){
        evidence.append(textNode("h3","Targetless methods · separate A2D2 data"));
        for(const row of this.data.targetless_evidence){evidence.append(textNode("p",`${row.method}: ${row.metric.grade.toUpperCase()} · held-out objective ${row.metric.holdout.toFixed(4)}`,"cloud-description"));}
        evidence.append(textNode("p","Different datasets and objectives: these scores cannot be ranked against board-centre RMSE. A2D2 inputs are camera-registered; no independent extrinsic accuracy is established.","cloud-description"));
      }
      this.panel.querySelector(".method-status").textContent=this.code==="live"?"Ready · press Run live fit or One iteration":"Measured data loaded · drag 3D plots, zoom or inspect observations";
      this.draw();
    }
    resetLive(){
      this.live=new CalibrexDemoSolver.LiveFit(this.data.live_problem,{
        scenario:this.panel.querySelector('[aria-label="Live observation condition"]').value,
        loss:this.panel.querySelector('[aria-label="Live fitting loss"]').value,
      });
      this.point=this.live.fitIndices[0];this.zoom=1;this.panel.querySelector(".evidence-zoom input").value="1";
      const notes={normal:"All 32 measured fit centres and 8 untouched holdouts. The start is a deliberately perturbed native reference.",limited:"Only 2 distinct measured fit captures are retained; 30 are excluded. The 8 holdouts stay unchanged. Rotation around the line joining the two centres is unobservable, so updates are blocked.",outliers:"4 fit targets are deliberately shifted by (+0.60, −0.40, +0.50) m. These are injected errors, not original measurements. The 8 holdouts are unchanged. Compare least squares with Huber."};
      this.panel.querySelector(".evidence-scenario-note").textContent=notes[this.live.scenario]+" Larger dots: active fit. Faded dots: excluded captures.";
      this.panel.querySelector(".method-status").textContent="Ready · press Run live fit or One iteration";
    }
    addControl(root,label,min,max,step,apply){
      const el=document.createElement("label"),input=document.createElement("input"),out=document.createElement("output");el.textContent=label+" ";input.type="range";input.min=min;input.max=max;input.step=step;input.value="0";input.setAttribute("aria-label",label);out.textContent="0";
      input.addEventListener("input",()=>{out.textContent=Number(input.value).toFixed(2);apply(Number(input.value));});el.append(input,out);root.append(el);
    }
    step(){
      try{const more=this.live.step();this.draw();this.panel.querySelector(".method-status").textContent=`Actual browser fit · ${this.live.status} · ${this.live.iteration} iterations`;return more;}
      catch(error){this.stop();this.panel.querySelector(".method-status").textContent=error.message;return false;}
    }
    metricCards(metrics){
      const root=this.panel.querySelector(".evidence-metrics");root.replaceChildren();
      for(const m of metrics){const card=document.createElement("div");card.append(textNode("small",m.label),textNode("strong",typeof m.value==="number"?`${m.value.toFixed(m.unit?3:0)} ${m.unit||""}`:m.value));root.append(card);}
    }
    draw(){
      if(!this.view)return;
      const ctx=this.context,w=this.canvas.width,h=this.canvas.height;ctx.fillStyle="#091524";ctx.fillRect(0,0,w,h);
      ctx.strokeStyle="#20354a";ctx.lineWidth=1;for(let x=40;x<w;x+=80){ctx.beginPath();ctx.moveTo(x,20);ctx.lineTo(x,h-20);ctx.stroke();}for(let y=40;y<h;y+=80){ctx.beginPath();ctx.moveTo(20,y);ctx.lineTo(w-20,y);ctx.stroke();}
      const legend=this.panel.querySelector(".evidence-legend");legend.replaceChildren();
      let count;
      if(this.view.kind==="lines"){
        const chart=this.view.chart;count=chart.x.length;
        this.lines(ctx,w,h,chart.x,chart.series);
        chart.series.forEach((s,i)=>{const e=textNode("span",s.name);e.style.color=COLOURS[i];legend.append(e);});
        this.panel.querySelector(".evidence-point-detail").textContent=`t = ${chart.x[this.point].toFixed(2)} s · `+chart.series.map(s=>`${s.name}: ${s.values[this.point].toFixed(2)} m/s`).join(" · ");
      }else{
        let clouds=this.view.clouds;
        if(this.code==="live"){
          clouds=[{name:"Camera targets · injected errors marked red",points:this.live.targets},{name:"Current transformed LiDAR",points:this.live.points()}];
          const history=this.live.history,last=history.at(-1);
          this.metricCards([{label:"Actual iteration",value:this.live.iteration},{label:this.live.injectedIndices.length?"Fit RMSE · includes injected errors":"Fit centre RMSE",value:last.fit,unit:"m"},{label:"Untouched holdout RMSE",value:last.holdout,unit:"m"},{label:"Geometry rank · before damping",value:`${this.live.geometry.rank} / 6`},{label:"Active fit / untouched holdout",value:`${this.live.fitIndices.length} / ${this.live.holdIndices.length}`}]);
          const quality=this.panel.querySelector(".evidence-quality");
          quality.dataset.warning=String(this.live.geometry.rank<6||this.live.injectedIndices.length>0);
          quality.textContent=this.live.geometry.rank<6?"Underconstrained: a unique 6-DoF fit is unavailable. No optimizer updates will run.":this.live.injectedIndices.length?"Injected outliers: numerical convergence can be biased. Compare the untouched holdout residuals.":"Geometry rank 6/6. This is a shape diagnostic, not an accuracy guarantee.";
          const c=this.panel.querySelector(".evidence-history"),hc=c.getContext("2d");hc.fillStyle="#091524";hc.fillRect(0,0,c.width,c.height);
          this.lines(hc,c.width,c.height,history.map(v=>v.iteration),[{name:"Fit",values:history.map(v=>v.fit)},{name:"Holdout",values:history.map(v=>v.holdout)}],false);
        }else if(this.code==="gnss"){
          const g=this.data.lever_geometry;
          const candidate=g.points.map((p,i)=>p.map((v,k)=>k===2?0:v-g.body_to_enu[i][k][0]*this.lever[0]-g.body_to_enu[i][k][1]*this.lever[1]));
          clouds=[...clouds,{name:"Candidate body path (geometry only)",points:candidate}];
        }
        count=clouds[0].points.length;
        const rotated=clouds.map(c=>({...c,points:c.points.map(p=>{
          if(this.view.kind==="radar")return [Math.cos(this.yaw)*p[0]-Math.sin(this.yaw)*p[1],Math.sin(this.yaw)*p[0]+Math.cos(this.yaw)*p[1],0,p[3]];
          return p;
        })}));
        this.scatter(rotated);
        clouds.forEach((c,i)=>{const e=textNode("span",c.name);e.style.color=COLOURS[i%COLOURS.length];legend.append(e);});
        const p=clouds[0].points[this.point];
        this.panel.querySelector(".evidence-point-detail").textContent=`Observation ${this.point+1} / ${count} · ${p.slice(0,3).map(v=>v.toFixed(3)).join(", ")} m`+(p.length===4?` · reported relative speed ${p[3].toFixed(2)} m/s`:"")+(this.code==="live"?` · ${this.live.fitIndices.includes(this.point)?"FIT":this.live.holdIndices.includes(this.point)?"HOLDOUT":"EXCLUDED"}${this.live.injectedIndices.includes(this.point)?" · INJECTED ERROR":""}`:"");
      }
      if(this.code!=="live")this.metricCards(this.view.metrics);
      const slider=this.panel.querySelector(".evidence-point-label input");slider.max=String(count-1);slider.value=String(this.point);
      this.canvas.setAttribute("aria-label",`${this.view.title}. ${count} measured observations. Selected observation ${this.point+1}.`);
    }
    lines(ctx,w,h,x,series,cursor=true){
      const values=series.flatMap(s=>s.values),xmin=x[0],xmax=x.at(-1),lo=Math.min(...values),hi=Math.max(...values),span=hi-lo||1;
      const px=t=>50+(t-xmin)/(xmax-xmin||1)*(w-90),py=v=>h-35-(v-lo+span*.1)/(span*1.2)*(h-70);
      series.forEach((s,i)=>{ctx.strokeStyle=COLOURS[i];ctx.lineWidth=2.5;ctx.beginPath();s.values.forEach((v,k)=>k?ctx.lineTo(px(x[k]),py(v)):ctx.moveTo(px(x[k]),py(v)));ctx.stroke();});
      ctx.fillStyle="#a7bfd4";ctx.font="13px monospace";ctx.fillText(`${hi.toFixed(2)}`,7,32);ctx.fillText(`${lo.toFixed(2)}`,7,h-26);ctx.fillText(`${xmin.toFixed(1)}`,50,h-8);ctx.fillText(`${xmax.toFixed(1)}`,w-80,h-8);
      if(cursor){ctx.strokeStyle="#ffffff99";ctx.beginPath();ctx.moveTo(px(x[this.point]),20);ctx.lineTo(px(x[this.point]),h-30);ctx.stroke();}
    }
    scatter(clouds){
      const ctx=this.context,w=this.canvas.width,h=this.canvas.height,radar=this.view.kind==="radar";
      const project=p=>{
        if(radar||this.code==="gnss")return [p[0],p[1]];
        const x=Math.cos(this.angle)*p[0]+Math.sin(this.angle)*p[2],z=-Math.sin(this.angle)*p[0]+Math.cos(this.angle)*p[2];
        return [x,Math.cos(this.pitch)*p[1]-Math.sin(this.pitch)*z];
      };
      const all=clouds.flatMap(c=>c.points.map(project)),xs=all.map(p=>p[0]),ys=all.map(p=>p[1]);
      const lo=[Math.min(...xs),Math.min(...ys)],hi=[Math.max(...xs),Math.max(...ys)];
      const selected=project(clouds[0].points[this.point]),centre=this.zoom>1?selected:[(lo[0]+hi[0])/2,(lo[1]+hi[1])/2];
      const scale=Math.min((w-100)/(hi[0]-lo[0]||1),(h-100)/(hi[1]-lo[1]||1))*this.zoom;
      const pixel=p=>{const q=project(p);return [w/2+(q[0]-centre[0])*scale,h/2-(q[1]-centre[1])*scale];};
      if(["comparison","handeye","live"].includes(this.code)){
        ctx.strokeStyle="#ffba7466";for(let i=0;i<clouds[0].points.length;i++){if(this.code==="live"&&!this.live.fitIndices.includes(i)&&!this.live.holdIndices.includes(i))continue;const a=pixel(clouds[0].points[i]),b=pixel(clouds[1].points[i]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}
      }
      clouds.forEach((c,k)=>{
        if(this.code==="gnss"){ctx.strokeStyle=COLOURS[k];ctx.lineWidth=2;ctx.beginPath();c.points.forEach((p,i)=>i?ctx.lineTo(...pixel(p)):ctx.moveTo(...pixel(p)));ctx.stroke();}
        for(let i=0;i<c.points.length;i++){const p=c.points[i],q=pixel(p);ctx.fillStyle=radar?`hsl(${200-175*Math.max(0,Math.min(1,(p[3]+10)/20))} 90% 65%)`:COLOURS[k%COLOURS.length];ctx.globalAlpha=this.code==="live"&&!this.live.fitIndices.includes(i)&&!this.live.holdIndices.includes(i)?.15:1;ctx.beginPath();ctx.arc(q[0],q[1],radar?5:this.code==="live"&&this.live.fitIndices.includes(i)?5:3.5,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;
          if(this.code==="live"&&k===0&&this.live.injectedIndices.includes(i)){ctx.strokeStyle="#ff657f";ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(q[0]-7,q[1]-7);ctx.lineTo(q[0]+7,q[1]+7);ctx.moveTo(q[0]-7,q[1]+7);ctx.lineTo(q[0]+7,q[1]-7);ctx.stroke();}}
      });
      const q=pixel(clouds[0].points[this.point]);ctx.strokeStyle="#fff";ctx.lineWidth=2;ctx.beginPath();ctx.arc(...q,9,0,Math.PI*2);ctx.stroke();
      ctx.fillStyle="#a7bfd4";ctx.font="13px monospace";ctx.fillText(`Horizontal extent ${(hi[0]-lo[0]).toFixed(2)} m · zoom ${this.zoom.toFixed(1)}×`,22,h-14);
    }
  }
  globalThis.CalibrexEvidence={CODES,validate,EvidenceDemo};
})();
