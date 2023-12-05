from collections import defaultdict
import math


import pygame
import numpy as np
from matplotlib.cm import ScalarMappable
import cv2



class SimVis:
    """
    Class to visualize scenarios.
    """
    def _on_screen_pos(self, p, screen_size):
        x = int(screen_size[0] * (p[0] - self._dims[0])/self._w)
        bany = screen_size[1] - self._banner_size
        y = screen_size[1] - int(bany * (p[1] - self._dims[2])/self._h)
        return (x, y)

    def __init__(self,
                 n_cell,
                 n_ues,
                 dims,
                 cmap='gist_rainbow',
                 ue_size=5,
                 cell_size=20):
        """
        dims is (x_min, x_max, y_min, y_max)
        cmap should be a string describing the matplotlib colormap type
        """
        pygame.init()
        self._cell_aspect = 4
        self._banner_size = 30
        self._white = (255,255,255)
        self._n_cell = n_cell
        self._n_ues = n_ues
        self._dims = dims
        self._w = dims[1] - dims[0]
        self._h = dims[3] - dims[2]
        self._ue_size= ue_size
        self._cell_size = cell_size
        self._cmap = ScalarMappable(cmap=cmap)
        self._cmap.set_clim(0, self._n_cell)
        self._font = pygame.font.SysFont(None, 24)
        self._small_font = pygame.font.SysFont(None, 16)

    def _color(self, value):
        rgba = self._cmap.to_rgba(value)
        return tuple([int(v*255) for v in rgba[0:3]])

    def _is_dark(self, c):
        """
        Determine if a color is dark, based on this blog
        https://www.nbdtech.com/Blog/archive/2008/04/27/Calculating-the-Perceived-Brightness-of-a-Color.aspx
        """
        b = math.sqrt(.241*c[0]**2 + .691*c[1]**2 + .068*c[2]**2)
        return b < 130

    def _render_single_cell(self, surf, color, left, top, loads=None):
        """
        Render a single cell square, starting at top_left corner.
        """
        r = pygame.Rect(left, top, self._cell_size*self._cell_aspect, self._cell_size)
        pygame.draw.rect(surf, color, r)
        if loads is not None:
            load_str = '(%.1f, %.1f)' % loads
            # Use white on dark colors.
            if self._is_dark(color):
                font_color = (255, 255, 255)
            else:
                font_color = (0, 0, 0)
            txt = self._small_font.render(load_str, True, font_color)
            surf.blit(txt, (left, top + 1))


    def _render_cell_stack(self, surf, x, y, stack_colors, stack_loads):
        top = int(y - len(stack_colors)*self._cell_size/2.)
        left = int(x - self._cell_size * self._cell_aspect/2.)
        for c, l in zip(stack_colors, stack_loads):
            self._render_single_cell(surf, c, left, top, l)
            top += self._cell_size

    def state_to_surface(self, state, nxt, surf) :
        """
        Render simulator state to the provided pygame surface.
        """
        # TODO: incorporate cell azimuth in visualization
        size = surf.get_size()
        surf.fill(self._white)
        if 'simulation_time' in state:
            now = state['simulation_time']
            text = self._font.render("Time " + str(int(now)) , True, (0, 128, 0))
            surf.blit(text, (10, 10))

        if 'reward' in state:
            txt = self._font.render("Reward: %s" % state['reward'], True, (0, 0, 0))
            surf.blit(txt, (300, 10))
        elif 'packet_loss' in state['UEs'] and 'demand' in state['UEs']:
            pl = state['UEs']['packet_loss']
            ue_demand = state['UEs']['packet_loss']
            total_loss = (pl * ue_demand).sum()
            txt = self._font.render("Total packet loss: %s" % (total_loss / ue_demand.sum()), True, (0, 0, 0))
            surf.blit(txt, (300, 10))

        demand = state['UEs'].get('demand', None)
        for ue_id, ss in enumerate(state['UEs']['serving_cell']):
            color = self._color(ss),
            pos = self._on_screen_pos(state['UEs']['position'][ue_id,:], size)
            pos1 = self._on_screen_pos(nxt['UEs']['position'][ue_id,:], size)

            if demand is None:
                ue_demand = 1.
            else:
                ue_demand = demand[ue_id]
            if ue_demand > 0 : # UE is active
                width = 0
            else :
                width = 1
            dx = pos1[0] - pos[0]
            dy = pos1[1] - pos[1]
            l = math.sqrt(dx * dx + dy * dy)
            if l == 0 :
                pygame.draw.circle(surf, color, pos, self._ue_size, width) # pay no attention to the documentation, vaily
            else :
                dx =  dx / l
                dy = dy / l
                raw = [(1.50*self._ue_size, 0), (-self._ue_size, self._ue_size/1), (-self._ue_size, -self._ue_size/1)]

                points = []
                for p in raw :
                    points.append((p[0]*dx - dy * p[1] + pos[0], p[0] * dy + p[1] * dx +pos[1]))
                pygame.draw.polygon(surf, color, points, width) # pay no attention to the documentation, vaily

        # Group cells into "stacks", i.e. cells that are in the same place physically.
        stacks = defaultdict(lambda: [[], []])
        for cell_id, pos in enumerate(state['cells']['position']):
            pos = self._on_screen_pos(pos, size)
            color = self._color(cell_id)
            if 'unsatisfied_load' in state['cells']:
                loads = (state['cells']['satisfied_load'][cell_id],
                         state['cells']['unsatisfied_load'][cell_id])
            elif 'demand' in state['cells'] and 'packet_loss' in state['cells']:
                demand = state['cells']['demand'][cell_id]
                pl = state['cells']['packet_loss'][cell_id]
                loads = (
                    demand * (1 - pl),
                    demand * pl
                )
            else:
                loads = None
            stacks[pos][0].append(color)
            stacks[pos][1].append(loads)

        for pos, stack in stacks.items():
            self._render_cell_stack(surf, *pos, *stack)

        return surf

    @staticmethod
    def bounds(states):
        """
        Get some bounds that include all the positions in all the states.
        Convenience method to be called to figure out dims for creating a SimVis object.
        """
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        for state in states:
            for p in np.concatenate((state['UEs']['position'], state['cells']['position'])):
                if p[0] < min_x:
                    min_x = p[0]
                if p[0] > max_x:
                    max_x = p[0]
                if p[1] < min_y:
                    min_y = p[1]
                if p[1] > max_y:
                    max_y = p[1]
        dx = 0.2 * (max_x - min_x)/2
        dy = 0.2 * (max_y - min_y)/2
        return [min_x - dx, max_x + dx, min_y - dy, max_y + dy]

    def height_px(self, width):
        """
        Get the height in pixels given a width (preserving aspect ratio)
        """
        return int(width*(self._h / self._w))

    def create_still(self, state, output_path, width_px=800, height_px=800):
        screen = pygame.Surface((width_px, height_px))
        self.state_to_surface(state, state, screen)
        frame = pygame.surfarray.array3d(screen)
        frame = cv2.flip(frame, 0)
        frame = np.rot90(frame, k=-1)
        cv2.imwrite(str(output_path), frame)

    def create_movie(self, states, output_path, width_px=800, height_px=800, max_duration=60):
        """
        Create a movie from a list of states, writing it to output path.
        """
        if len(states) < 2:
            raise ValueError("There should be at least two states to make a movie")
        frame_rate = 1./(states[1]['simulation_time'] - states[0]['simulation_time'])
        duration = states[-1]['simulation_time'] - states[0]['simulation_time']
        if (max_duration is not None) and (duration > max_duration):
            frame_rate = frame_rate * duration / max_duration

        screen = pygame.Surface((width_px, height_px))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width_px, height_px))
        for pos, state in enumerate(states):
            if pos == (len(states)-1) :
                nxt = state
            else :
                nxt = states[pos+1]
            screen = self.state_to_surface(state, nxt, screen)
            frame = pygame.surfarray.array3d(screen)
            frame = cv2.flip(frame, 0)
            frame = np.rot90(frame, k=-1)
            out.write(frame)
        out.write(frame)
        out.release()

