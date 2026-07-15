#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg_plotter.py – 실시간 g‑g 다이어그램 (동심원 & 꼬리, 수정판)
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

G_CONST = 9.81
MAX_R_G = 1.5
CIRCLE_STEP = 0.25
TAIL_LENGTH = 400
UPDATE_HZ = 30


class GGPlotterFancy(Node):
    def __init__(self):
        super().__init__('gg_plotter_fancy')
        self.axg, self.ayg = [], []

        # ── Matplotlib ─────────────────────────────────────────────
        plt.style.use('dark_background')
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.max_r = MAX_R_G
        self._draw_concentric_circles()

        self.scat = self.ax.scatter([], [], s=10, c='deepskyblue', alpha=0.5)
        self.head, = self.ax.plot([], [], 'o', ms=10,
                                  mfc='crimson', mec='white', mew=0.5)

        # 🔧 수정된 부분: x축과 y축 라벨을 서로 바꿈
        self.ax.set_xlabel('Lateral g')
        self.ax.set_ylabel('Longitudinal g')
        self.ax.set_title('G‑G Diagram', pad=14)
        self.ax.set_aspect('equal', adjustable='box')
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # ── ROS2 ───────────────────────────────────────────────────
        self.create_subscription(Imu, '/imu/data', self.imu_cb, 10)
        self.create_timer(1.0 / UPDATE_HZ, self.update_plot)
        self.get_logger().info('g‑g 플로터(동심원) 실행 중 – 창을 닫거나 Ctrl‑C 로 종료')

    def imu_cb(self, msg: Imu):
        # 데이터 수집 순서는 그대로 유지 (axg: longitudinal, ayg: lateral)
        self.axg.append(msg.linear_acceleration.x / G_CONST)
        self.ayg.append(msg.linear_acceleration.y / G_CONST)
        if len(self.axg) > TAIL_LENGTH:
            del self.axg[:-TAIL_LENGTH]
            del self.ayg[:-TAIL_LENGTH]

        r_now = np.hypot(self.axg[-1], self.ayg[-1])
        if r_now > self.max_r:
            self.max_r = np.ceil(r_now / CIRCLE_STEP) * CIRCLE_STEP
            self._draw_concentric_circles()

    def update_plot(self):
        if not self.axg:
            return
            
        # 🔧 수정된 부분: column_stack에서 ayg(lateral)를 x축으로, axg(longitudinal)를 y축으로 설정
        pts = np.column_stack((self.ayg, self.axg))
        self.scat.set_offsets(pts)

        N = len(pts)
        self.scat.set_sizes(np.linspace(20, 5, N))
        alphas = np.linspace(1.0, 0.05, N)
        colors = np.column_stack((np.repeat([[0.0, 0.75, 1.0]], N, 0), alphas))
        self.scat.set_facecolors(colors)
        self.scat.set_alpha(None)

        # 🔧 수정된 부분: head의 x, y 데이터 순서를 바꿈
        self.head.set_data([self.ayg[-1]], [self.axg[-1]])

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _draw_concentric_circles(self):
        for child in [c for c in self.ax.get_children() if isinstance(c, Circle)]:
            child.remove()
        for r in np.arange(CIRCLE_STEP, self.max_r + 1e-3, CIRCLE_STEP):
            self.ax.add_patch(Circle((0, 0), r, fill=False, lw=0.6,
                                     ls='--', color='gray', alpha=0.4))
            if r % (CIRCLE_STEP * 2) < 1e-3:
                p = r / np.sqrt(2)
                self.ax.text(p, p, f'{r:.2g} g', fontsize=7, color='gray')
        self.ax.set_xlim(-self.max_r, self.max_r)
        self.ax.set_ylim(-self.max_r, self.max_r)
        self.fig.canvas.draw_idle()


def main():
    rclpy.init()
    node = GGPlotterFancy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        print('[gg_plotter] 종료')


if __name__ == '__main__':
    main()