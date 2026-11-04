// `?screenshot` in the address opens the portrait map alone, for the README picture: no page around it,
// labels at reading size, the nodes packed round them to fill the window, and no intro. The recorder
// (web/scripts/screenshot.mjs) steps the map frame by frame through `window.atlasCapture`.
import type { AtlasCapture } from './atlas/atlasEngine';
import './screenshot.css';

declare global {
  interface Window {
    atlasCapture?: AtlasCapture;
  }
}

export const SCREENSHOT = new URLSearchParams(window.location.search).has('screenshot');

if (SCREENSHOT) document.documentElement.setAttribute('data-screenshot', '');
