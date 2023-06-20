import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import plotly.graph_objects as go


parser = argparse.ArgumentParser(
    description="Generate potential energy surface"
)

parser.add_argument( '--prefix', type=str, help='Prefix to add to the beginning of logged files', dest='prefix', default='', required=False, nargs='?', const='')
parser.add_argument( '--load-dir', type=str, help='Directory from which to load the data', dest='load_dir', required=True)
parser.add_argument( '--n-components', type=int, help='Number of dimensions for SHEAP plot. Must be 2 or 3', dest='n_components', choices=[2,3], default=2)
parser.add_argument( '--scale', type=float, help='Scaling factor for SHEAP volumes', dest='scale', default=1)

args = parser.parse_args()

def main():

    input_fname = os.path.join(args.load_dir, args.prefix+f'sheap-{args.n_components}d.xyz')


    if args.n_components == 2:
        data = np.loadtxt(input_fname, skiprows=2, usecols=(1, 2, 10, 8))  # x, y, size, color

        fig, ax = plt.subplots(figsize=(16, 8))

        ax.set_title('SHEAP')

        tmp = data.copy()

        x = tmp[:, 0]
        y = tmp[:, 1]
        s = tmp[:, 2]
        c = tmp[:, 3]

        mnx = 1.1*np.min(x)
        mxx = 1.1*np.max(x)

        mny = 1.1*np.min(y)
        mxy = 1.1*np.max(y)

        ax.set_aspect('equal')

        ax.set_xlim([mnx, mxx])
        ax.set_ylim([mny, mxy])

        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        w, h = bbox.width, bbox.height

        pix_per_plot_unit = w*fig.dpi/abs(mxx-mnx)

        s *= pix_per_plot_unit
        s  = np.pi*(s**2)
        s *= args.scale

        sc = ax.scatter(x, y, s=s, c=c)

        cbar = fig.colorbar(sc, fraction=0.018)
        cbar.set_label(r'$E_{DFT}$', rotation=0, fontsize=14, labelpad=20)

        plt.savefig(
            os.path.join(args.load_dir, args.prefix+'sheap-2d.png'),
            bbox_inches='tight'
        )
    else:
        data = np.loadtxt(input_fname, skiprows=2, usecols=(1, 2, 3, 11, 9)) # x, y, z, size, color

        x = data[:, 0].copy()
        y = data[:, 1].copy()
        z = data[:, 2].copy()
        s = data[:, 3].copy()
        c = data[:, 4].copy()

        s *= args.scale

        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=s,
                sizemode='diameter',
                color=c,                # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=1.0,
                colorbar=dict(
                    title="Colorbar"
                ),
                line=dict(
                    width=1,
                )
            ),
        )])

        # tight layout
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(aspectmode='data')
        )

        fig.write_image(os.path.join(args.load_dir, args.prefix+'sheap-3d.png'))

    print("Saving/loading data to/from:", args.load_dir)


if __name__ == '__main__':
    main()
