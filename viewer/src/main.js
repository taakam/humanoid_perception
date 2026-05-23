import * as THREE from "three";
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
import { CSS2DRenderer, CSS2DObject } from "three/examples/jsm/renderers/CSS2DRenderer.js";

const semanticScene = new THREE.Scene();

const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(window.innerWidth, window.innerHeight);
labelRenderer.domElement.style.position = "absolute";
labelRenderer.domElement.style.top = "0px";
labelRenderer.domElement.style.pointerEvents = "none";
document.body.appendChild(labelRenderer.domElement);

function makeLabel(text) {
  const div = document.createElement("div");
  div.style.color = "white";
  div.style.background = "rgba(0,0,0,0.75)";
  div.style.padding = "3px 6px";
  div.style.borderRadius = "5px";
  div.style.fontSize = "12px";
  div.style.whiteSpace = "nowrap";
  div.innerHTML = text;
  return new CSS2DObject(div);
}

function addSemanticMarker(obj) {
  if (!obj.has_3d_association || !obj.centroid_xyz) return;

  const [x, y, z] = obj.centroid_xyz;

  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(0.04, 16, 16),
    new THREE.MeshBasicMaterial({ color: 0xff3333 })
  );
  sphere.position.set(x, y, z);
  semanticScene.add(sphere);

  const affordances = obj.affordances ? obj.affordances.join(", ") : "";
  const label = makeLabel(`<b>${obj.object_id}</b><br>${affordances}`);
  label.position.set(x, y, z + 0.08);
  semanticScene.add(label);
}

const viewer = new GaussianSplats3D.Viewer({
  threeScene: semanticScene,
  cameraUp: [0, -1, 0],
  initialCameraPosition: [-1, -4, 6],
  initialCameraLookAt: [0, 4, 0],
});

fetch("/data/semantic_objects_3d.json")
  .then((res) => res.json())
  .then((objects) => {
    console.log(`Loaded ${objects.length} semantic objects`);
    objects.forEach(addSemanticMarker);
  });

viewer.addSplatScene("/data/point_cloud.ply", {
  splatAlphaRemovalThreshold: 5,
  showLoadingUI: true,
})
.then(() => {
  viewer.start();

  function renderLabels() {
    requestAnimationFrame(renderLabels);
    labelRenderer.render(semanticScene, viewer.getCamera());
  }

  renderLabels();
});