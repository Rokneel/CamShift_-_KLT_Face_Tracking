% Clear workspace
clc; clear;

% Create cascade object detector using vision toolbox
faceDetector = vision.CascadeObjectDetector();

% Analyze a frame from given video and use cascadeObjectDetector
videoReader = VideoReader('faceDataset/CameraShift.mp4');
videoFrame = readFrame(videoReader);
bbox = step(faceDetector, videoFrame);

% Draw bounding box on detected face
videoOutput = insertObjectAnnotation(videoFrame,'rectangle',bbox,'Face');
figure, imshow(videoOutput), title('Face Detected');

% First bounding box turn into list of 4 points
bboxPoints = bbox2points(bbox(1,:));

% Detect features within bounding box
points = detectMinEigenFeatures(im2gray(videoFrame), "ROI", bbox);

% Output found points
figure, imshow(videoFrame), hold on, title("Features detected");
plot(points);

% Initialize point tracker and set initial points
pointTracker = vision.PointTracker("MaxBidirectionalError", 2);
points = points.Location;
initialize(pointTracker, points, videoFrame);

% Initialize video player for displaying output
videoPlayer = vision.VideoPlayer("Position",...
    [100 100 [size(videoFrame,2), size(videoFrame, 1)]+ 30]);

% Initialize previous points and bounding box for IOU calculation
oldPoints = points;
prevBbox = bbox;

% Create IOU variable
iou = [];

% Start timer for processing time 
tic 

while hasFrame(videoReader)
    % Get the next frame
    videoFrame = readFrame(videoReader);

    % Track the points
    [points, isFound] = step(pointTracker, videoFrame);
    visiblePoints = points(isFound, :);
    oldInliers = oldPoints(isFound, :);

    if size(visiblePoints, 1) >= 2
        % Estimate geometric transformation between old points and new
        % points to get rid of outlying points
        [xform, inlierIdx] = estimateGeometricTransform2D(...
            oldInliers, visiblePoints, "similarity", "MaxDistance", 4);
        oldInliers = oldInliers(inlierIdx, :);
        visiblePoints = visiblePoints(inlierIdx, :);

        % Use transformation to calculate new bounding box points
        bboxPoints = transformPointsForward(xform, bboxPoints);

        % Calculate IOU
        if ~isempty(prevBbox)
            iou(end+1) = bboxOverlapRatio(prevBbox, bbox);
        end

        % Update previous bounding box
        prevBbox = bbox;

        % Draw a bounding box around the face
        bboxPolygon = reshape(bboxPoints', 1,[]);
        videoFrame = insertShape(videoFrame, "polygon", bboxPolygon, ...
            "LineWidth", 2);

        % Output tracked points
        videoFrame = insertMarker(videoFrame, visiblePoints, "+", ...
            "Color", "white");

        % Reset points 
        oldPoints = visiblePoints;
        setPoints(pointTracker, oldPoints);
    end

    % Output modified video frame 
    step(videoPlayer, videoFrame);
end

% Release the video player
release(videoPlayer);


fprintf('Mean IOU: %f\n', mean(iou));
% Stop