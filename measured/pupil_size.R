# Apparent pupil diameter (mm) in the constricted and the dilated condition, the check that the luminance
# manipulation worked. Slope plot: one line per eye from constricted to dilated. One figure per camera, columns =
# Left | Right eye. White circle = mean with ± SD whiskers, labeled with both values. Color = condition
# (yellow = constricted, navy = dilated). Run with cwd = measured/.
Sys.setlocale("LC_CTYPE", "en_US.UTF-8")
library(tidyverse)

# Eye-level camera first, the order the paper reports the cameras in.
camera_labels <- c(cam2 = "Eye-level camera", cam1 = "Bottom-left camera")
camera_file_names <- c(cam2 = "eye_level_camera", cam1 = "bottom_left_camera")

df <- read_csv("output/pupil_size.csv", show_col_types = FALSE) %>%
  rename(pupil_mm = apparent_pupil_diameter_mm) %>%
  mutate(
    camera = factor(camera, levels = names(camera_labels), labels = camera_labels),
    condition = factor(condition, levels = c("constricted", "dilated")),
    eye = factor(recode(eye, left_eye = "Left eye", right_eye = "Right eye"),
                 levels = c("Left eye", "Right eye"))
  )

cond_colors <- c(constricted = "#F4D03F", dilated = "#2C3E50")

summary_stats <- df %>%
  group_by(camera, eye, condition) %>%
  summarize(mean_mm = mean(pupil_mm), sd_mm = sd(pupil_mm), .groups = "drop") %>%
  mutate(
    mean_label = sprintf("%.2f ± %.2f", mean_mm, sd_mm),
    position = as.numeric(condition) + if_else(condition == "constricted", -0.15, 0.15),
    text_position = as.numeric(condition) + if_else(condition == "constricted", -0.34, 0.34)
  )

camera_plot <- function(camera_label) {
  camera_df <- filter(df, camera == camera_label)
  camera_stats <- filter(summary_stats, camera == camera_label)
  ggplot(camera_df, aes(x = condition, y = pupil_mm)) +
    geom_line(aes(group = participant), color = "gray70", linewidth = 0.6, alpha = 0.8) +
    geom_point(aes(fill = condition, group = participant), size = 2.2, shape = 21, color = "black", stroke = 0.2) +
    geom_errorbar(data = camera_stats,
                  aes(x = position, y = mean_mm, ymin = mean_mm - sd_mm, ymax = mean_mm + sd_mm, group = 1),
                  width = 0.08, linewidth = 0.7, color = "#063D51", inherit.aes = FALSE) +
    geom_point(data = camera_stats, aes(x = position, y = mean_mm), inherit.aes = FALSE,
               size = 2.5, shape = 21, fill = "#FFFFFF", color = "#063D51", stroke = 0.8) +
    geom_text(data = camera_stats, aes(x = text_position, y = mean_mm, label = mean_label),
              inherit.aes = FALSE, vjust = 0.5, hjust = 0.5, size = 3.5, angle = 90) +
    facet_grid(. ~ eye) +
    scale_fill_manual(values = cond_colors, guide = "none") +
    scale_x_discrete(labels = c(constricted = "Constricted", dilated = "Dilated")) +
    scale_y_continuous(expand = expansion(mult = 0.15)) +
    labs(x = "Pupil condition", y = "Apparent pupil diameter (mm)") +
    theme_minimal() +
    theme(
      strip.text = element_text(size = 12),
      panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA)
    )
}

dir.create("output", showWarnings = FALSE)
for (camera_id in names(camera_labels)) {
  path <- sprintf("output/pupil_size_%s.png", camera_file_names[[camera_id]])
  ggsave(path, camera_plot(camera_labels[[camera_id]]), width = 6.0, height = 3.4, dpi = 600,
         device = ragg::agg_png)
  cat("wrote", path, "\n")
}
