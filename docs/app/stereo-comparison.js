"use strict";
(() => {
  const MODELS=["kb4","mei","ds"],LABELS=["KB4","MEI","Double Sphere"];
  function mismatchProfile(data){
    const points=data?.correspondences,n=data?.matches;
    if(!Array.isArray(points)||!Number.isInteger(n)||n<1||points.length!==n)throw Error("Invalid comparison matches");
    const values=points.map(p=>{
      const a=p?.left?.[1],b=p?.right?.[1];if(!Number.isFinite(a)||!Number.isFinite(b))throw Error("Invalid comparison rows");return Math.abs(b-a);
    }).sort((a,b)=>a-b);
    const counts=Array(12).fill(0);let overflow=0;
    for(const v of values){if(v>=3)overflow++;else counts[Math.floor(v/.25)]++;}
    return {counts,percentages:counts.map(v=>100*v/n),overflow,overflow_pct:100*overflow/n,median_px:(values[Math.floor((n-1)/2)]+values[Math.floor(n/2)])/2,n};
  }
  function comparisonProfiles(frames,code){
    if(!Array.isArray(frames)||frames.length!==3)throw Error("Need all three models");
    const first=frames[0];
    if(!["000","014","027"].includes(code)||typeof first?.raw!=="string"||!Array.isArray(first.image_size)||first.image_size.length!==2||!first.image_size.every(v=>Number.isInteger(v)&&v>0)||!Number.isFinite(first.focal_px)||first.focal_px<=0)throw Error("Invalid comparison camera");
    return frames.map((d,i)=>{
      if(d?.camera_model!==MODELS[i]||d.frame!==code||d.capture_kind!=="real"||d.depth_truth!==null||d.raw!==first.raw||JSON.stringify(d.image_size)!==JSON.stringify(first.image_size)||!Number.isFinite(d.focal_px)||Math.abs(d.focal_px-first.focal_px)>1e-9)throw Error("Mismatched comparison photograph or output camera");
      return mismatchProfile(d);
    });
  }
  class StereoComparison {
    constructor(root,getFrame,onInspect){
      this.root=root;this.getFrame=getFrame;this.onInspect=onInspect;this.sequence=0;this.frame=null;this.active="kb4";this.data=null;this.loading=false;
      root.innerHTML=`<label class="stereo-comparison-toggle"><input type="checkbox" aria-label="Compare fisheye models side by side">Compare all three models side by side</label><div class="stereo-comparison-body" hidden><div class="stereo-comparison-heading"><h3>One photograph. Three models.</h3><p>Rectified left images with saved guides. Each model has its own accepted match set; these diagnostics do not rank depth accuracy.</p></div><p class="stereo-comparison-status" role="status"></p><button type="button" class="stereo-comparison-retry" hidden>Retry model comparison</button><div class="stereo-comparison-cards"></div><p class="stereo-comparison-note">Vertical mismatch: |right y − left y|. Equal 0.25 px bins; every chart uses 0–3 px and 0–100% of that model’s accepted matches. Values ≥ 3 px are counted separately. Correspondence consistency is not independent depth truth.</p></div>`;
      this.toggle=root.querySelector("input");this.body=root.querySelector(".stereo-comparison-body");this.cards=root.querySelector(".stereo-comparison-cards");this.status=root.querySelector(".stereo-comparison-status");this.retry=root.querySelector("button");
      this.toggle.addEventListener("change",()=>{
        this.body.hidden=!this.toggle.checked;
        if(this.toggle.checked)this.load();else{this.sequence++;this.loading=false;this.cards.replaceChildren();}
      });
      this.retry.addEventListener("click",()=>this.load(true));
    }
    setFrame(code,active){
      const changed=code!==this.frame;this.frame=code;this.active=active;
      if(changed){this.sequence++;this.loading=false;this.data=null;this.cards.replaceChildren();}
      if(!this.toggle.checked)return;
      if(this.data)this.updateActive();else if(!this.loading)this.load();
    }
    async load(refresh=false){
      if(!this.toggle.checked||!this.frame)return;
      if(this.data){this.render();return;}
      const seq=++this.sequence,code=this.frame;this.loading=true;this.cards.replaceChildren();this.retry.hidden=true;this.status.textContent=`Loading all three models for photographed pair ${code}…`;
      try{
        const data=await Promise.all(MODELS.map(m=>this.getFrame(m,code,refresh))),profiles=comparisonProfiles(data,code);
        const images=await Promise.all(data.map(async d=>{const image=new Image();image.src=d.rectified;await image.decode();if(image.naturalWidth!==2*d.image_size[0]||image.naturalHeight!==d.image_size[1])throw Error("Invalid comparison image size");return image;}));
        if(seq!==this.sequence||!this.toggle.checked||code!==this.frame)return;
        this.data=data;this.profiles=profiles;this.images=images;this.loading=false;this.render();
      }catch(error){if(seq===this.sequence&&this.toggle.checked){this.loading=false;this.cards.replaceChildren();this.status.textContent=`Could not load comparison for pair ${code}: ${error.message}`;this.retry.hidden=false;}}
    }
    render(){
      if(!this.data||!this.toggle.checked)return;
      this.cards.replaceChildren();this.retry.hidden=true;
      this.data.forEach((d,i)=>{
        const model=MODELS[i],p=this.profiles[i],card=document.createElement("article");card.className="stereo-comparison-card";card.dataset.model=model;card.classList.toggle("is-active",model===this.active);card.setAttribute("aria-label",`${LABELS[i]} comparison for photographed pair ${this.frame}`);
        card.innerHTML=`<h4>${LABELS[i]}<small>${i===0?"Upstream calibration":"Fit to measured corners"}</small></h4><canvas class="stereo-comparison-image" width="${d.image_size[0]}" height="${d.image_size[1]}" role="img"></canvas><dl><div><dt>Accepted matches</dt><dd>${p.n}</dd></div><div><dt>Median vertical mismatch</dt><dd>${p.median_px.toFixed(3)} <small>px</small></dd></div></dl><div class="stereo-comparison-axis"><span>100% of accepted matches</span></div><div class="stereo-comparison-histogram" role="img"></div><div class="stereo-comparison-axis"><span>0 px</span><span>1</span><span>2</span><span>3 px</span></div><p class="stereo-comparison-overflow">${p.overflow} / ${p.n} at ≥ 3 px</p><button type="button" aria-pressed="${model===this.active}">Inspect ${LABELS[i]}</button>`;
        const canvas=card.querySelector("canvas");canvas.getContext("2d").drawImage(this.images[i],0,0,...d.image_size,0,0,...d.image_size);canvas.setAttribute("aria-label",`${LABELS[i]}, pair ${this.frame}, saved rectified left image with epipolar guides.`);
        const histogram=card.querySelector(".stereo-comparison-histogram");
        p.percentages.forEach((value,j)=>{const bar=document.createElement("span");bar.style.height=value+"%";bar.title=`${(j*.25).toFixed(2)}–${((j+1)*.25).toFixed(2)} px: ${p.counts[j]} / ${p.n} (${value.toFixed(1)}%)`;histogram.append(bar);});
        histogram.setAttribute("aria-label",`Absolute vertical mismatch, 0–3 pixels, common 0–100 percent scale. ${p.counts.map((n,j)=>`${(j*.25).toFixed(2)} to ${((j+1)*.25).toFixed(2)} pixels: ${n} matches`).join("; ")}. At least 3 pixels: ${p.overflow} matches.`);
        card.querySelector("button").addEventListener("click",()=>this.onInspect(model));this.cards.append(card);
      });
      this.updateActive();
    }
    updateActive(){
      for(const card of this.cards.children){const active=card.dataset.model===this.active;card.classList.toggle("is-active",active);card.querySelector("button").setAttribute("aria-pressed",String(active));}
      const text=`Photographed pair ${this.frame} · all three models loaded · same raw photographs and output focal length · selected ${LABELS[MODELS.indexOf(this.active)]}`;
      if(this.status.textContent!==text)this.status.textContent=text;
    }
  }
  globalThis.CalibrexStereoComparison={mismatchProfile,comparisonProfiles,StereoComparison};
})();
