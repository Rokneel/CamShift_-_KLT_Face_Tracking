%Clear workspace
clc; clear;

%Create cascade object dectector using vision toolbox.
faceDetector = vision.CascadeObjectDetector();

%Analyse a frame from given video and use cascadeObjectDetector
videoFileReader = VideoReader('faceDataset/CameraShift.mp4');
videoFrame = readFrame(videoFileReader);
bbox = step(faceDetector, videoFrame);

%Draw bounding box on detected face.
videoOutput = insertObjectAnnotation(videoFrame,'rectangle',bbox,'Face');
figure, imshow(videoOutput), title('Face Detected');

[hueChannel,~,~] = rgb2hsv(videoFrame);

%Output Hue Channel data and draw bounding box around detected face.
figure, imshow(hueChannel), title('Hue channel data');
rectangle('Position', bbox(1,:),'LineWidth',2,'EdgeColor',[1 1 0])

%feature based Detector 
noseFound = vision.CascadeObjectDetector('Nose', 'UseROI', true);
noseBoundingBox = step(noseFound, videoFrame, bbox(1,:));

%tracker created using Histograms
Htracker = vision.HistogramBasedTracker;

%initialize histogram tracking using Hue channel data from the nose
%detection.
initializeObject(Htracker, hueChannel, noseBoundingBox(1,:));

%Display Frames from video being used.
PlayVideo = vision.VideoPlayer;

%Initialize previous bounding bbox for IOU calculation
prevBbox = bbox;

%Create IOU variable
iou = [];

%face tracked over video frames, continues till video is finished.
while hasFrame(videoFileReader)
    
    %Analyse next video frame
    videoFrame = readFrame(videoFileReader);

    % RGB coverted to HSV
    [hueChannel,~,~] = rgb2hsv(videoFrame);

    %data extracted from Hue channel is tracked
    bbox = step(Htracker, hueChannel);

    %Calculate IOU
    iou(end+1) = bboxOverlapRatio(prevBbox, bbox);

    %Update previous bounding box
    prevBbox = bbox;

    %Draw a bounding box on the face being tracked
    videoOutput = insertObjectAnnotation(videoFrame,'rectangle',bbox,'Face');

    %Output the modified video frame 
    step(PlayVideo, videoOutput);
end

%Release Video
release(PlayVideo)

% Output IOU results 
fprintf('Mean IOU: %f\n', mean(iou));
fprintf('Median IOU: %f\n', median(iou));