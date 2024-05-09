from nilearn import plotting


def create_map(stat_img):
    # fig = plotting.plot_surf_stat_map(
    #     fsaverage.infl_right, texture, hemi='right',
    #     title='Surface right hemisphere', colorbar=True,
    #     threshold=1., bg_map=curv_right_sign,
    # )
    # fig.show()

    fig  = plotting.plot_stat_map(
    stat_img,
    threshold=3,
    title="plot_stat_map",
    cut_coords=[36, -27, 66],
    radiological=True,
    )
    fig.show()