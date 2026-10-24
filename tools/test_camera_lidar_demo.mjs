import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import "../docs/app/camera-lidar-demo.js";
const {validate,project}=globalThis.CalibrexCameraLidar;
const read=frame=>JSON.parse(fs.readFileSync(new URL(`../docs/app/samples/camera_lidar_${frame}.json`,import.meta.url),"utf8"));

test("real KITTI projections match independent SciPy reference poses",()=>{
  for(const frame of ["0060","0090"]){
    const data=validate(read(frame));
    assert.deepEqual(data.image_size,[1242,375]);
    assert.ok(data.points.length>1000);
    for(const ref of data.reference_projections){
      ref.uvz.forEach((expected,i)=>{
        const actual=project(data.points[i],data.intrinsics,ref.offsets);
        expected.forEach((v,k)=>assert.ok(Math.abs(actual[k]-v)<1e-7,`${frame}/${i}/${k}`));
      });
    }
    const zero=[0,0,0,0,0,0],point=data.points[0];
    const vendor=project(point,data.intrinsics,zero);
    const shifted=project(point,data.intrinsics,[0,0,0,.1,0,0]);
    assert.ok(Math.abs(shifted[0]-vendor[0]-data.intrinsics[0]*.1/point[2])<1e-8);
    assert.notDeepEqual(project(point,data.intrinsics,[0,3,0,0,0,0]),vendor);
    for(const change of [{capture_kind:"synthetic"},{points:[[Infinity,0,1]]},{intrinsics:[0,1,1,1]},{image_size:[-1,375]},{frame:1}])assert.throws(()=>validate({...data,...change}),/Invalid/);
  }
});
test("points behind the adjusted camera cannot become visible overlays",()=>{
  assert.equal(project([1,1,-1],[700,700,600,180],[0,0,0,0,0,0]),null);
  assert.equal(project([1,1,1],[700,700,600,180],[0,0,0,0,0,-2]),null);
});
