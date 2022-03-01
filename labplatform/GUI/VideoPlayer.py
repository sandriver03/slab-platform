from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
import PIL, PIL.TiffImagePlugin
import io
import matplotlib.animation as animation

__all__ = ['videofig']


def videofig(num_frames, redraw_func, play_fps=25, big_scroll=30, key_func=None,
             grid_specs=None, layout_specs=None, figname=None, save_dir=None, format='tiff',
             *args):
    """
    Figure with horizontal scrollbar and play capabilities
    This script is mainly inspired by the elegant work of João Filipe Henriques
        https://www.mathworks.com/matlabcentral/fileexchange/29544-figure-to-play-and-analyze-videos-with-custom-plots-on-top?focused=5172704&tab=function
    :param num_frames: an integer, number of frames in a sequence
    :param redraw_func: callable with signature redraw_func(f, axes)
                      used to draw a new frame at position f using axes, which is a instance of Axes class in matplotlib
    :param play_fps: an integer, number of frames per second, used to control the play speed
    :param big_scroll: an integer, big scroll number used when pressed page down or page up keys.
    :param key_func: optional callable which signature key_func(key), used to provide custom key shortcuts.
    :param grid_specs: optional dictionary, used to specify the gridspec of the main drawing pane.
                     For example, grid_specs = {'nrows': 2, 'ncols': 2} will create a gridspec with 2 rows and 2 cols.
    :param layout_specs: optional list, used to specify the layout of the gridspec of the main drawing pane.
                     For example, layout_specs = ['[:, 0]', '[:, 1]'] means:
                        gs = ... Some code to create the main drawing pane gridspec ...
                        ax1 = plt.subplot(gs[:, 0])
                        ax2 = plt.subplot(gs[:, 1])
    :param save_dir: a string, used to specify the directory to which figures will be saved.
    :param save_dir: str, image format to be saved
    :param args: other optional arguments
    :return: None
    """
    # Check arguments
    check_int_scalar(num_frames, 'num_frames')
    check_callback(redraw_func, 'redraw_func')
    check_int_scalar(play_fps, 'play_fps')
    check_int_scalar(big_scroll, 'big_scroll')
    if key_func:
        check_callback(key_func, 'key_func')

    # Initialize figure
    if not figname:
        figname = 'VideoPlayer'
    fig_handle = plt.figure(num=figname)

    # We do not want to show the slider when saving figures
    slider_ratio = 3 if save_dir is None else 1e-3

    # We use GridSpec to support embedding multiple plots in the main drawing pane.
    # A nested grid demo can be found at https://matplotlib.org/users/plotting/examples/demo_gridspec06.py
    # Construct outer grid, which contains the main drawing pane and slider ball
    outer_grid = gridspec.GridSpec(2, 1,
                                   left=0, bottom=0, right=1, top=1,
                                   height_ratios=[97, slider_ratio], wspace=0.0, hspace=0.0)

    # Construct inner grid in main drawing pane
    if grid_specs is None:
        grid_specs = {'nrows': 1, 'ncols': 1}  # We will have only one axes by default

    inner_grid = gridspec.GridSpecFromSubplotSpec(subplot_spec=outer_grid[0], **grid_specs)
    if layout_specs:
        # eval() can't work properly in list comprehension which is inside a function.
        # Refer to: http://bugs.python.org/issue5242
        # Maybe I should find another way to implement layout_specs without using eval()...
        axes_handle = []
        for spec in layout_specs:
            axes_handle.append(plt.Subplot(fig_handle, eval('inner_grid' + spec)))
    else:
        num_inner_plots = grid_specs['nrows'] * grid_specs['ncols']
        axes_handle = [plt.Subplot(fig_handle, inner_grid[i]) for i in range(num_inner_plots)]

    for ax in axes_handle:
        fig_handle.add_subplot(ax)

    if len(axes_handle) == 1:
        axes_handle = axes_handle[0]
        axes_handle.set_axis_off()

    # Build scrollbar
    scroll_axes_handle = plt.Subplot(fig_handle, outer_grid[1])
    scroll_axes_handle.set_facecolor('lightgoldenrodyellow')
    fig_handle.add_subplot(scroll_axes_handle)
    scroll_handle = Slider(scroll_axes_handle, '', 0.0, num_frames - 1, valinit=0.0)

    def draw_new(_):
        redraw_func(int(scroll_handle.val), axes_handle)
        fig_handle.canvas.draw_idle()

    def scroll(new_f):
        new_f = min(max(new_f, 0), num_frames - 1)  # clip in the range of [0, num_frames - 1]
        cur_f = scroll_handle.val

        # Stop player at the end of the sequence
        if new_f == (num_frames - 1):
            play.running = False

        if cur_f != new_f:
            # move scroll bar to new position
            scroll_handle.set_val(new_f)

        return axes_handle

    def play(period):
        play.running ^= True  # Toggle state
        if play.running:
            frame_idxs = range(int(scroll_handle.val), num_frames)
            play.anim = FuncAnimation(fig_handle, scroll, frame_idxs,
                                      interval=1000 * period, repeat=False, blit=False)
            plt.draw()
        else:
            play.anim.event_source.stop()

    # Set initial player state
    play.running = False

    def key_press(event):
        key = event.key
        f = scroll_handle.val
        if key == 'left':
            scroll(f - 1)
        elif key == 'right':
            scroll(f + 1)
        elif key in ['pageup', 'up']:
            scroll(f - big_scroll)
        elif key in ['pagedown', 'down']:
            scroll(f + big_scroll)
        elif key == 'home':
            scroll(0)
        elif key == 'end':
            scroll(num_frames - 1)
        elif key == 'enter':
            play(1 / play_fps)
        elif key == 'backspace':
            play(5 / play_fps)
        else:
            if key_func:
                key_func(key)

    # Register events
    scroll_handle.on_changed(draw_new)
    fig_handle.canvas.mpl_connect('key_press_event', key_press)

    if save_dir is None:
        # Draw initial frame
        redraw_func(0, axes_handle)

        # Start playing
        play(1 / play_fps)

        # plt.show() has to be put in the end of the function,
        # otherwise, the program simply won't work, weird...
        plt.show()
    else:
        print('=> Saving plots to: {}...'.format(save_dir))
        os.makedirs(save_dir, exist_ok=True)

        # Due to the issue of https://github.com/matplotlib/matplotlib/issues/7614
        # we need to save the first image twice. Otherwise, the first saved image will be distorted.
        fname_prefix = figname
        filehandle = None
        for index in list(range(0, num_frames)):
            redraw_func(index, axes_handle)
            fig_handle.canvas.draw_idle()
            if format == 'jpg':
                if not isinstance(axes_handle, list):
                    if matplotlib.get_backend() == 'nbAgg':
                        print('Warning: remove "%matplotlib notebook", otherwise saved images will be distorted!')
                        extent = axes_handle.get_window_extent().transformed(fig_handle.dpi_scale_trans.inverted())
                    else:
                        extent = None
                fig_handle.savefig(os.path.join(save_dir, fname_prefix + '_{:04d}.jpg'.format(index)),
                                   bbox_inches=extent)
            elif format in ('tiff', 'tif'):
                if filehandle is None:
                    fname = os.path.join(save_dir, fname_prefix + '.tiff')
                    filehandle = PIL.TiffImagePlugin.AppendingTiffWriter(fname)
                # save matplotlib figure to a in-memory buffer
                buf = io.BytesIO()
                fig_handle.savefig(buf, format='tiff')
                # read this buffer in to a PIL image object
                frame = PIL.Image.open(buf)
                # create new frame in the file tiff file and write current frame
                filehandle.newFrame()
                frame.save(filehandle)
                # close in memory buffer and PIL image
                buf.close()
                frame.close()

            else:
                raise ValueError('the image format: {} cannot be saved'.format(format))

        # close the tiff file if applicable
        if filehandle is not None:
            filehandle.close()


def check_int_scalar(a, name):
    assert isinstance(a, int), '{} must be a int scalar, instead of {}'.format(name, type(name))


def check_callback(a, name):
    # Check http://stackoverflow.com/questions/624926/how-to-detect-whether-a-python-variable-is-a-function
    # for more details about python function type detection.
    assert callable(a), '{} must be callable, instead of {}'.format(name, type(name))


def play_images(image_data, title='', nchannels=1, play_channel=0, frame_info=None,
                display_frame_Nr=True, play_fps=20, big_scroll=30, key_func=None,
                grid_specs=None, layout_specs=None, cmap=None, clim=None, colorbar=True,
                save_dir=None):
    """
    Args:
        image_data: image data, 3-D numpy array
        title: title of the image window
        nchannels: number of different channels in the image data
        play_channel: which channel to be played
        frame_info: array of frame information, in form of [frame_nr, frame_tm]
        display_frame_Nr: if display frame number on the image
        cmap: str, colormap used to display images. see matplotlib.pyplot.get_cmap; or a custom color map
        clim: tuple, (vmin, vmax) of the color scale to be set
        colorbar: bool, if display color bar

        play_fps:
        big_scroll:
        key_func:
        grid_specs:
        layout_specs:
        save_dir:
        see function videofig
    """
    if frame_info is None:
        frame_info = np.arange(1, image_data.shape[0] + 1).reshape(-1, 1)
    # pick the images to be played by channel
    chs = play_channel if nchannels > 0 else 0
    idx = np.where(frame_info[:, 0] % nchannels == chs)[0]
    img_to_play = image_data[idx, :, :]
    img_to_play_frameNr = frame_info[idx, 0]
    # display location
    if display_frame_Nr:
        display_pos = (0.85*image_data.shape[1], 0.15*image_data.shape[2])
    # vmin, vmax
    if clim is not None:
        vmin, vmax = clim
    else:
        vmin, vmax = None, None
    # color map
    if cmap is not None:
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

    def redraw_fn(f, axes):
        if not redraw_fn.initialized:
            redraw_fn.img = axes.imshow(img_to_play[0], cmap=cmap, vmin=vmin, vmax=vmax)
            redraw_fn.initialized = True
            if display_frame_Nr:
                redraw_fn.text = axes.text(display_pos[0], display_pos[1],
                                           img_to_play_frameNr[0])
            if colorbar:
                axes.figure.colorbar(redraw_fn.img, ax=axes)
        else:
            redraw_fn.img.set_data(img_to_play[f])
            if display_frame_Nr:
                redraw_fn.text.set_text(img_to_play_frameNr[f])

    # run the GUI
    redraw_fn.initialized = False
    videofig(img_to_play.shape[0], redraw_fn, figname=title, play_fps=play_fps,
             big_scroll=big_scroll, key_func=key_func, grid_specs=grid_specs,
             layout_specs=layout_specs, save_dir=save_dir)


if __name__ == '__main__':

    import numpy as np

    def redraw_fn(f, axes):
        amp = float(f) / 3000
        f0 = 3
        t = np.arange(0.0, 1.0, 0.001)
        s = amp * np.sin(2 * np.pi * f0 * t)
        if not redraw_fn.initialized:
            redraw_fn.l, = axes.plot(t, s, lw=2, color='red')
            redraw_fn.initialized = True
        else:
            redraw_fn.l.set_ydata(s)

    redraw_fn.initialized = False
    videofig(100, redraw_fn)