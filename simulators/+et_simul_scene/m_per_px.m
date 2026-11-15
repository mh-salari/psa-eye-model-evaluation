function scale = m_per_px(c, eye)
%  Meters at the eye per image pixel, at the eye's distance from the camera.
    scale = norm(c.trans(1:3, 4) - eye.position) / c.focal_length;
end
