%Clear workspace
clc; clear;

%Create cascade object dectector using vision toolbox.
faceDetector = vision.CascadeObjectDetector();

%Analyse a frame from given video and use cascadeObjectDetector
videoReader = VideoReader('faceDataset/Smile.mp4');
videoFrame = readFrame(videoReader);
bbox = step(faceDetector, videoFrame);

%Draw bounding box on detected face.
videoOutput = insertObjectAnnotation(videoFrame,'rectangle',bbox,'Face');
figure, imshow(videoOutput), title('Face Detected');

%First bounding box turn into list of 4 points
bboxPoints = bbox2points(bbox(1,:));

points = detectMinEigenFeatures(im2gray(videoFrame), "ROI", bbox);

%Output found points
figure, imshow(videoFrame), hold on, title("Features detected");
plot(points);

pointTracker = vision.PointTracker("MaxBidirectionalError", 2);

%Intialize tracker with initial point locations and the initial video
%frame.
points = points.Location;


initialize(pointTracker, points, videoFrame);

videoPlayer = vision.VideoPlayer("Position",...
    [100 100 [size(videoFrame,2), size(videoFrame, 1)]+ 30]);

oldPoints = points;

% Start timer for processing time 
tic 

while hasFrame(videoReader)
    %get the next frame
    videoFrame = readFrame(videoReader);

    %Track the points.
    [points, isFound] = step(pointTracker, videoFrame);
    visiblePoints = points(isFound, :);
    oldInliers = oldPoints(isFound, :);

    if size(visiblePoints, 1) >= 2 
        %Estrimate geometric transformation between olf points and new
        %points to get rid of outlying points
        [xform, inlierIdx] = estimateGeometricTransform2D(...
            oldInliers, visiblePoints, "similarity", "MaxDistance", 4);
        oldInliers = oldInliers(inlierIdx, :);
        visiblePoints = visiblePoints(inlierIdx, :);

        %Use transformation to bounding box points
        bboxPoints = transformPointsForward(xform, bboxPoints);

        %draw a bounding box around the face
        bboxPolygon = reshape(bboxPoints', 1,[]);
        videoFrame = insertShape(videoFrame, "polygon", bboxPolygon, ...
            "LineWidth", 2);

        %Output tracked points
        videoFrame = insertMarker(videoFrame, visiblePoints, "+", ...
            "Color", "white");

        %Reset points 
        oldPoints = visiblePoints;
        setPoints(pointTracker, oldPoints);
        
    end
    %Output modified video frame 
    step(videoPlayer, videoFrame);
end

%Stop timing the processing time and display processing time
processingTime = toc;
disp("Processing time: " + processingTime + " seconds");

release(videoPlayer);