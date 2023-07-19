from Image_check import *

# Get all the images
high_img = get_corrected_img(all_path2, 'high')
rot_high_img = get_corrected_img(all_path1, 'high')

low_img = get_corrected_img(all_path2, 'low')
rot_low_img = get_corrected_img(all_path1, 'low')

plot_1D_profile(high_img)
plot_1D_profile(low_img)

plot_2D_profile(high_img)
plot_2D_profile(low_img)


# Sava all the images generated in this way
# sub_img = np.subtract(rot_high_img, high_img)
# high_prev = get_corrected_img(prev_all_path, 'high')
# dark = get_image(dark_path, 'high')

# find_structures(high_prev, rot_high_img)
#
# plot_1D_profile(high_img)
# plot_2D_profile(high_img[0:100, 0:100])
#
# plot_histogram(high_img[4000:4734, :], 'High gain non rotated check')
# plot_histogram(high_prev[4000:4734, :], 'High gain image previous check')
# plot_histogram(rot_high_img[4000:4734, :], 'High gain image rotated check')
#
#
# mean_high = 12496.726
# mean_high_rotated = 12385.41
# mean_high_prev = 10176.54
#
# find_std(high_img[4000:4734, :], mean_high)
# find_std(high_prev[4000:4734, :], mean_high_prev)
# find_std(rot_high_img[4000:4734, :], mean_high_rotated)


# rotate_hig_img = np.divide(rotate_hig_img, ratio)
# Rotate the images
# soft_rot_high = np.rot90(rotate_hig_img)
save_image(high_img, main_path, "Nonrotated_high")
save_image(rot_high_img, main_path, "rotate_high_img")
# save_image(high_prev, main_path, 'Previous_high_image')
# save_image(sub_img, main_path, 'subtracted_image')


# high_part = high_img[4000:4734, :]
# rot_high_part = rot_high_img[4000:4734, :]
# high_prev_part = high_prev[4000:4734, :]
# save_image(high_part, main_path, "part of high image")
# save_image(rot_high_part, main_path, "part of rotational high image")
# save_image(high_prev_part, main_path, "part of high previous image")

# save_image(ratio, main_path, 'ratio image')
# save_image(low_prev, main_path, "previous_low")
#
# # high_img = high_img.astype(np.int16, casting="same_kind")
# # rotate_hig_img = rotate_hig_img.astype(np.int16, casting="same_kind")
# diff_img = np.subtract(rotate_hig_img, high_img)
# save_image(diff_img, main_path, "subtracted image")
# # get_mean_hist(high_prev, high_img, rotate_hig_img)
# # get_mean_hist(low_prev, low_img, rotate_low_img)
# dark = get_image(dark_path, "high")
# save_image(dark, main_path, "dark image-high")
# # m, s = get_mean_sections(low_prev)
# # print(len(m))
# # print(m)
