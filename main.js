import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

class HumanDetector {
  constructor() {
    this.video = document.getElementById('video');
    this.canvas = document.getElementById('output-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.cameraHeight = document.getElementById('camera-height');
    this.cameraAngle = document.getElementById('camera-angle');
  }

  async init() {
    // Load the COCO-SSD model
    this.model = await cocoSsd.load();
    
    // Setup video stream
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 640,
          height: 480
        }
      });
      this.video.srcObject = stream;
    } catch (err) {
      console.error('Error accessing camera:', err);
    }

    // Start detection loop
    this.detectFrame();
  }

  estimateDistance(y2, frameHeight) {
    const cameraHeight = parseFloat(this.cameraHeight.value);
    const cameraAngle = parseFloat(this.cameraAngle.value) * Math.PI / 180;
    
    // Convert pixel position to angle from camera
    const angleInFrame = Math.atan2((frameHeight/2 - y2), 1000); // 1000 is approximate focal length
    const totalAngle = cameraAngle + angleInFrame;
    
    // Calculate distance using trigonometry
    return cameraHeight / Math.tan(totalAngle);
  }

  async detectFrame() {
    // Detect objects in the video frame
    const predictions = await this.model.detect(this.video);
    
    // Clear previous drawings
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
    // Process each prediction
    predictions.forEach(prediction => {
      if (prediction.class === 'person') {
        const [x, y, width, height] = prediction.bbox;
        const distance = this.estimateDistance(y + height, this.canvas.height);
        
        // Draw bounding box
        this.ctx.strokeStyle = '#00ff00';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(x, y, width, height);
        
        // Draw distance label
        this.ctx.fillStyle = '#00ff00';
        this.ctx.font = '16px Arial';
        this.ctx.fillText(`Distance: ${distance.toFixed(2)}m`, x, y - 5);
      }
    });
    
    // Continue detection loop
    requestAnimationFrame(() => this.detectFrame());
  }
}

// Initialize the detector when the page loads
window.onload = async () => {
  const detector = new HumanDetector();
  await detector.init();
};