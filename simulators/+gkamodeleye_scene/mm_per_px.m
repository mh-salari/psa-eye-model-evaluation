function scale = mm_per_px(sg)
%  mm at the eye per image pixel, at the eye's distance from the camera.
    scale = norm(sg.cameraPosition.translation) / sg.cameraIntrinsic.matrix(1, 1);
end
