function anatomy = population_anatomy(s)
%  gkaModelEye's own eye in the fitted parameters: the corneal apex radius of its default mean corneal power (mm), and
%  its aperture stop centered on the optical axis.
    anatomy.anterior_radius = 1000 * (s.keratometric_index - 1) / mean(s.default_kvals);
    anatomy.off_axis_nasal_mm = 0;
    anatomy.off_axis_superior_mm = 0;
end
