import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import "../docs/app/method-gallery.js";
const {METHODS,validateStereo,stereoCrop}=globalThis.CalibrexMethods;
const read=(p)=>JSON.parse(fs.readFileSync(new URL("../docs/app/samples/"+p,import.meta.url),"utf8"));

test("gallery exposes eleven distinct demos and recorded sources remain attributable",()=>{
  assert.equal(METHODS.length,11);assert.equal(new Set(METHODS.map(m=>m.id)).size,11);
  for(const m of METHODS.filter(m=>m.image)) {
    assert.ok(m.source.startsWith("../assets/"));assert.ok(m.license);assert.ok(m.dataset);
    assert.ok(fs.existsSync(new URL("../docs/"+m.image.slice(3),import.meta.url)));
  }
});
test("each native stereo profile has valid images and finite metric receipts",()=>{
  const index=read("fisheye_stereo_demo.json");
  assert.equal(index.schema,"calibrex.fisheye_stereo_demo/v1");
  assert.deepEqual(index.scene.depth_planes_m,[3,6,12]);
  assert.ok(index.method.includes("No truth depths enter matching"));
  for(const code of ["kb4","mei","ds"]){
    const model=validateStereo(read(index.models[code].url));
    assert.ok(model.matches>20);assert.equal(model.baseline_m,.3);
    assert.ok(model.vertical_median_px<.2);assert.ok(model.median_relative_depth_error<.03);
    assert.throws(()=>validateStereo({...model,raw:"javascript:alert(1)"}),/Invalid/);
    assert.throws(()=>validateStereo({...model,vertical_median_px:NaN}),/Invalid/);
    assert.throws(()=>validateStereo({...model,matches:-1}),/Invalid/);
    assert.equal(model.correspondences.length,model.matches);
    const point=model.correspondences[0];
    for(const change of [
      {left:[-1,20]}, {right:[320,20]}, {disparity_px:0},
      {depth_m:point.depth_m+1}, {right:[point.right[0]+1,point.right[1]]},
    ])assert.throws(()=>validateStereo({...model,correspondences:[{...point,...change},...model.correspondences.slice(1)]}),/Invalid/);
    assert.throws(()=>validateStereo({...model,correspondences:[]}),/Invalid/);
  }
});
test("the default fisheye gallery uses real photographs without fabricated depth truth",()=>{
  const index=read("fisheye_real_demo.json");
  assert.equal(index.capture_kind,"real");
  assert.deepEqual(Object.keys(index.frames),["000","014","027"]);
  for(const [frame,receipt] of Object.entries(index.frames)){
    const data=validateStereo(read(receipt.url));
    assert.equal(data.capture_kind,"real");assert.equal(data.frame,frame);
    assert.equal(data.depth_truth,null);assert.equal(data.median_relative_depth_error,undefined);
    assert.deepEqual(data.image_size,[640,400]);assert.ok(data.matches>100);
    assert.ok(data.baseline_m>.099&&data.baseline_m<.100);
    assert.throws(()=>validateStereo({...data,depth:"data:text/html;base64,SGVsbG8="}),/Invalid/);
  }
  const source=fs.readFileSync(new URL("../docs/app/method-gallery.js",import.meta.url),"utf8");
  assert.ok(source.includes('this.json("samples/fisheye_real_demo.json?'));
  assert.ok(!source.includes('this.json("samples/fisheye_stereo_demo.json")'));
});
test("MEI and Double Sphere use their own real-corner fits on the same photograph",()=>{
  const index=read("fisheye_real_demo.json"),kb4=read(index.frames["000"].url);
  for(const model of ["mei","ds"]){
    const data=validateStereo(read(index.model_frames[model]["000"].url));
    assert.equal(data.capture_kind,"real");assert.equal(data.camera_model,model);
    assert.equal(data.raw,kb4.raw);assert.notEqual(data.rectified,kb4.rectified);
    assert.equal(data.depth_truth,null);assert.ok(data.fit_excluded_reprojection_px<1);
    assert.ok(data.matches>100);assert.ok(data.vertical_median_px<1);
  }
});


test("stereo close-ups preserve subpixel centres and never cross a camera boundary",()=>{
  for(const p of [[0,0],[10.25,3.75],[639.75,399.5],[320,200]]){
    const g=stereoCrop(p,640,400);
    assert.equal(g.size,41);
    assert.ok(g.sx>=0&&g.sy>=0&&g.sx+g.sw<=640&&g.sy+g.sh<=400);
    assert.ok(g.dx>=0&&g.dy>=0&&g.dx+g.sw<=41&&g.dy+g.sh<=41);
    assert.equal(g.sx+g.cx-g.dx-.5,p[0]);
    assert.equal(g.sy+g.cy-g.dy-.5,p[1]);
  }
  assert.deepEqual(stereoCrop([0,0],640,400),{sx:0,sy:0,sw:21,sh:21,dx:20,dy:20,cx:20.5,cy:20.5,size:41});
  assert.equal(stereoCrop([639,399],640,400).sw,21);
  for(const p of [[NaN,0],[-1,0],[640,0],[0,400],null,[1]])assert.throws(()=>stereoCrop(p,640,400),/Invalid/);
  assert.throws(()=>stereoCrop([0,0],0,400),/Invalid/);
  // Exercise every measured match, including fractional right-image rows.
  const index=read("fisheye_real_demo.json");
  for(const receipt of [...Object.values(index.frames),...Object.values(index.model_frames).flatMap(Object.values)]){
    const data=read(receipt.url);
    for(const p of data.correspondences)for(const side of ["left","right"]){
      const g=stereoCrop(p[side],...data.image_size);
      assert.ok(Math.abs(g.sx+g.cx-g.dx-.5-p[side][0])<1e-10);
      assert.ok(Math.abs(g.sy+g.cy-g.dy-.5-p[side][1])<1e-10);
    }
  }
});
