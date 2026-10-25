"use strict";
(() => {
  const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
  const xyz=p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite);
  function residualMatrix(data){
    if(data?.id!=="comparison"||data.capture_kind!=="real"||!Array.isArray(data.views)||!data.views.length)throw Error("Invalid board comparison");
    const problem=data.live_problem,ids=data.views[0]?.result?.holdout_frame_ids,fit=data.views[0]?.result?.train_frame_ids;
    const unique=a=>Array.isArray(a)&&a.length>0&&a.every(v=>typeof v==="string"&&v.length>0)&&new Set(a).size===a.length;
    if(!unique(ids)||!unique(fit)||!unique(problem?.ids)||!unique(problem?.fit_ids)||!Array.isArray(problem.target)||problem.target.length!==problem.ids.length||!problem.target.every(xyz)||!same([...fit].sort(),[...problem.fit_ids].sort())||ids.length+fit.length!==problem.ids.length||ids.some(id=>fit.includes(id))||[...ids,...fit].some(id=>!problem.ids.includes(id)))throw Error("Invalid shared board split");
    const source=new Map(problem.ids.map((id,i)=>[id,problem.target[i]])),seen=new Set();
    const rows=data.views.map(view=>{
      if(typeof view.id!=="string"||!view.id||seen.has(view.id)||typeof view.label!=="string"||!same(view.result?.holdout_frame_ids,ids)||!same(view.result?.train_frame_ids,fit)||!Array.isArray(view.clouds)||view.clouds.length!==2||!view.clouds.every(c=>Array.isArray(c.points)&&c.points.length===ids.length&&c.points.every(xyz)))throw Error("Mismatched method observations or split");
      seen.add(view.id);
      const cells=ids.map((id,i)=>{
        const camera=view.clouds[0].points[i],aligned=view.clouds[1].points[i];
        if(camera.some((v,k)=>Math.abs(v-source.get(id)[k])>1e-7))throw Error("Capture ID does not match camera observation");
        const deltaMm=aligned.map((v,k)=>(v-camera[k])*1000);
        return {id,deltaMm,valueMm:Math.hypot(...deltaMm)};
      });
      return {id:view.id,label:view.label,cells,rmseMm:Math.sqrt(cells.reduce((s,c)=>s+c.valueMm**2,0)/cells.length)};
    });
    return {ids:ids.slice(),fitCount:fit.length,rows,maxMm:Math.max(...rows.flatMap(r=>r.cells.map(c=>c.valueMm)))};
  }
  const node=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
  class BoardResiduals {
    constructor(root,onInspect){
      this.root=root;this.onInspect=onInspect;root.hidden=true;
      root.innerHTML=`<div class="board-residual-heading"><span class="eyebrow">SAME MEASURED CAPTURES / SAVED NATIVE RESULTS</span><h3></h3><p class="board-residual-split"></p></div><div class="board-residual-scroll" tabindex="0" role="region" aria-label="Board residual matrix; scroll horizontally on narrow screens"><table aria-label="Held-out board-centre residuals in millimetres"></table></div><div class="board-residual-legend"><span>0 mm</span><i></i><span class="board-residual-max"></span></div><output class="board-residual-detail"></output><p class="board-residual-note">Click a cell to inspect the same capture in the 3D plot. Arrow keys move across methods and captures; the observation slider stays linked. Colours share one scale. Values are 3D centre residuals from the displayed rounded coordinates; the side panel retains the saved native evaluation metric, formatted for display. Methods optimize different objectives. This is held-out consistency, without independent extrinsic truth. A2D2 targetless scores use separate data.</p>`;
      this.table=root.querySelector("table");this.key=null;
      this.table.addEventListener("keydown",e=>{
        if(!this.data||!["ArrowLeft","ArrowRight","ArrowUp","ArrowDown","Home","End"].includes(e.key))return;
        const button=e.target.closest("button");if(!button)return;e.preventDefault();
        let r=Number(button.dataset.row),c=Number(button.dataset.col);
        if(e.key==="ArrowLeft")c--;if(e.key==="ArrowRight")c++;if(e.key==="ArrowUp")r--;if(e.key==="ArrowDown")r++;
        if(e.key==="Home")c=0;if(e.key==="End")c=this.data.ids.length-1;
        r=Math.max(0,Math.min(this.data.rows.length-1,r));c=Math.max(0,Math.min(this.data.ids.length-1,c));
        this.onInspect(this.data.rows[r].id,c);this.buttons[r][c].focus();
      });
    }
    clear(){this.data=null;this.key=null;this.root.hidden=true;this.table.replaceChildren();this.root.querySelector("output").textContent="";}
    setData(data){
      this.clear();this.data=residualMatrix(data);const d=this.data;this.buttons=[];
      this.root.querySelector("h3").textContent=`${d.rows.length} methods. ${d.ids.length} held-out captures.`;
      this.root.querySelector(".board-residual-split").textContent=`Shared split: ${d.fitCount} fit / ${d.ids.length} holdout captures. Matrix values: ||transformed LiDAR centre − camera centre||, millimetres.`;
      const head=node("thead"),line=node("tr");line.append(node("th","Native method"));this.headers=[];
      d.ids.forEach(id=>{const th=node("th",id.split("-").at(-1));th.scope="col";th.title=id;line.append(th);this.headers.push(th);});line.append(node("th","RMS · mm"));head.append(line);this.table.append(head);
      const body=node("tbody");this.rowHeaders=[];
      d.rows.forEach((row,r)=>{
        const tr=node("tr"),label=node("th",row.label);label.scope="row";tr.append(label);this.rowHeaders.push(label);
        this.buttons[r]=row.cells.map((cell,c)=>{
          const td=node("td"),button=node("button",cell.valueMm.toFixed(2));button.type="button";button.dataset.row=String(r);button.dataset.col=String(c);button.tabIndex=-1;
          const t=d.maxMm?cell.valueMm/d.maxMm:0;button.style.backgroundColor=`rgb(${Math.round(17+92*t)} ${Math.round(72-5*t)} ${Math.round(74-33*t)})`;
          button.setAttribute("aria-label",`${row.label}, ${cell.id}: ${cell.valueMm.toFixed(3)} millimetres. Inspect held-out capture ${c+1} of ${d.ids.length}.`);
          button.addEventListener("click",()=>this.onInspect(row.id,c));td.append(button);tr.append(td);return button;
        });
        const rms=node("td",row.rmseMm.toFixed(2));rms.className="board-residual-rms";tr.append(rms);body.append(tr);
      });
      this.table.append(body);this.root.querySelector(".board-residual-max").textContent=d.maxMm.toFixed(2)+" mm";this.root.hidden=false;
    }
    select(method,index){
      const d=this.data,r=d?.rows.findIndex(row=>row.id===method);if(!d||r<0||!Number.isInteger(index)||index<0||index>=d.ids.length)return;
      const key=method+":"+index;if(key===this.key)return;this.key=key;
      this.buttons.forEach((row,i)=>row.forEach((button,j)=>{const active=i===r&&j===index;button.setAttribute("aria-pressed",String(active));button.tabIndex=active?0:-1;button.classList.toggle("is-capture",j===index);}));
      this.headers.forEach((h,j)=>h.classList.toggle("is-capture",j===index));this.rowHeaders.forEach((h,i)=>h.classList.toggle("is-method",i===r));
      const row=d.rows[r],cell=row.cells[index];this.root.querySelector("output").textContent=`${row.label} · ${cell.id} · 3D residual ${cell.valueMm.toFixed(3)} mm · ΔX ${cell.deltaMm[0].toFixed(3)} / ΔY ${cell.deltaMm[1].toFixed(3)} / ΔZ ${cell.deltaMm[2].toFixed(3)} mm`;
    }
  }
  globalThis.CalibrexBoardResiduals={residualMatrix,BoardResiduals};
})();
