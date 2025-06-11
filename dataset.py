###############################################################################
#  VIOPreprocessedDataset -- revised 2025-06-07
#  * ENU-aligned IMU axes (swap X↔Y, flip Z)            «axis-fix»
#  * Correlated IMU noise (AR(1), ρ = 0.8)              «imu-noise»
#  * Optional motion-blur + random mask on images       «img-aug»
###############################################################################

import os, random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as tv
from PIL import Image, ImageFilter
from scipy.spatial.transform import Rotation as R


def _pil_to_tensor_resize(img: Image.Image, wh: Tuple[int, int]) -> torch.Tensor:
    img = img.resize((wh[0], wh[1]), Image.BILINEAR)
    return tv.functional.pil_to_tensor(img).float().div_(255.0)


class VIOPreprocessedDataset(data.Dataset):
    def __init__(
            self,
            data_root: str,
            imu_csv: str,
            odom_csv: str,
            sequence_length: int = 6,
            imu_per_frame: int = 50,
            image_size: Tuple[int, int] = (320, 240),
            imu_noise_std: float = 0.01,
            imu_dropout_prob: float = 0.3,
            first_frame_mode: str = "extrapolate",
            p_invert: float = 0.10,
            p_bgr: float = 0.10,
            p_gray: float = 0.10,
            p_mask: float = 0.25,
            p_motion: float = 0.30,  # NEW ▸ probability of motion-blur
            seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.N = sequence_length
        self.M = imu_per_frame
        self.W, self.H = image_size
        self.noise_std = imu_noise_std
        self.drop_p = imu_dropout_prob
        self.p_invert = p_invert
        self.p_bgr = p_bgr
        self.p_gray = p_gray
        self.p_mask = p_mask
        self.p_motion = p_motion
        self.first_mode = first_frame_mode.lower()
        assert self.first_mode in {"zero", "extrapolate", "global"}

        # ───────────── load image paths & timestamps ─────────────
        ts = np.load(os.path.join(data_root, "image_timestamps.npy")).astype(np.int64)
        img_dir = os.path.join(data_root, "images")
        img_paths = np.array([os.path.join(img_dir, f"img_{i:06d}.png") for i in range(len(ts))])
        ok = np.vectorize(os.path.exists)(img_paths)
        self.img_paths, self.img_ts = img_paths[ok], ts[ok]
        assert len(self.img_paths) >= self.N
        self.seq_starts = np.arange(len(self.img_paths) - self.N + 1, dtype=np.int32)

        # ───────────── IMU & odometry CSVs ───────────────────────
        imu_arr = pd.read_csv(imu_csv).values.astype(np.float32)
        self.imu_ts = imu_arr[:, 0].astype(np.int64)
        self.imu_data = imu_arr[:, 1:]  # (Tᵢ, 6)

        odo_arr = pd.read_csv(odom_csv).values.astype(np.float32)
        self.odo_ts = odo_arr[:, 0].astype(np.int64)
        self.odo_xyz = odo_arr[:, 1:4]
        self.odo_rpy = odo_arr[:, 4:7]

        # pre-compute interp. indices & α
        idx_r = np.clip(np.searchsorted(self.odo_ts, self.img_ts, side="left"),
                        1, len(self.odo_ts) - 1)
        self.idx_L, self.idx_R = idx_r - 1, idx_r
        denom = (self.odo_ts[self.idx_R] - self.odo_ts[self.idx_L]).astype(np.float64)
        denom[denom == 0] = 1
        self.alpha = ((self.img_ts - self.odo_ts[self.idx_L]).astype(np.float64) / denom).astype(np.float32)

    # ─────────────────────────────────────────────────────────────
    #  Augment helpers
    # ─────────────────────────────────────────────────────────────
    def _motion_blur(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p_motion:
            return img
        k = random.choice([3, 5, 7])
        return img.filter(ImageFilter.BoxBlur(k / 2))

    def _mask_pil(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p_mask:
            return img
        W, H = img.size
        area = random.uniform(0.05, 0.20) * W * H
        ratio = random.uniform(0.3, 3.3)
        mw, mh = int(np.sqrt(area / ratio)), int(np.sqrt(area * ratio))
        x0, y0 = random.randint(0, max(1, W - mw)), random.randint(0, max(1, H - mh))
        img2 = img.copy();
        img2.paste((0, 0, 0), (x0, y0, x0 + mw, y0 + mh))
        return img2

    def _augment_imu(self, arr: np.ndarray) -> np.ndarray:
        # # axis-fix: swap ax↔ay, gx↔gy & flip az,gz
        # arr = arr.copy()
        # arr[:, [0, 1]] = arr[:, [1, 0]]
        # arr[:, [3, 4]] = arr[:, [4, 3]]
        # arr[:, 2] *= -1;
        # arr[:, 5] *= -1

        # correlated noise
        if self.noise_std > 0:
            noise = np.random.randn(*arr.shape).astype(np.float32) * self.noise_std
            for t in range(1, len(noise)):
                noise[t] = 0.8 * noise[t - 1] + 0.2 * noise[t]
            arr += noise

        # dropout block
        if random.random() < self.drop_p:
            blk = random.randint(5, min(10, self.M))
            start = random.randint(0, self.M - blk)
            arr[start:start + blk] = 0.0
        return arr

    # ─────────────────────────────────────────────────────────────
    #  Dataset interface
    # ─────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.seq_starts)

    def __getitem__(self, idx):
        s = self.seq_starts[idx]
        fr = slice(s, s + self.N)
        ts_seq = self.img_ts[fr]

        # pick one colour-mode for this sequence
        p = random.random()
        mode = "invert" if p < self.p_invert else \
            "bgr" if p < self.p_invert + self.p_bgr else \
                "gray" if p < self.p_invert + self.p_bgr + self.p_gray else "none"

        # ────────────── images ───────────────────────────────────
        img_tensors = []
        for pth in self.img_paths[fr]:
            img = Image.open(pth).convert("RGB")
            if mode == "invert":
                img = Image.fromarray(255 - np.asarray(img))
            elif mode == "bgr":
                img = Image.fromarray(np.asarray(img)[..., ::-1])
            elif mode == "gray":
                img = tv.functional.rgb_to_grayscale(img, num_output_channels=3)
            img = self._motion_blur(img)
            img = self._mask_pil(img)
            img_tensors.append(_pil_to_tensor_resize(img, (self.W, self.H)))
        imgs = torch.stack(img_tensors)  # (N, 3, H, W)

        # ────────────── IMU batch ────────────────────────────────
        imu_batch = []
        for t in ts_seq:
            end = np.searchsorted(self.imu_ts, t, side="left")
            start = max(0, end - self.M)
            sl = self.imu_data[start:end]
            if sl.shape[0] < self.M:  # pad / repeat first row
                pad = np.repeat(sl[:1], self.M - sl.shape[0], axis=0) if sl.size else np.zeros((self.M, 6), np.float32)
                sl = np.vstack((pad, sl))
            elif sl.shape[0] > self.M:
                sl = sl[-self.M:]
            imu_batch.append(torch.from_numpy(self._augment_imu(sl)))
        imu_batch = torch.stack(imu_batch)  # (N, M, 6)

        # ────────────── Pose interpolation & relative Δ ─────────
        L, R = self.idx_L[fr], self.idx_R[fr]
        a = self.alpha[fr][:, None]
        xyz = self.odo_xyz[L] + (self.odo_xyz[R] - self.odo_xyz[L]) * a
        d_ang = (self.odo_rpy[R] - self.odo_rpy[L] + np.pi) % (2 * np.pi) - np.pi
        ang = self.odo_rpy[L] + d_ang * a
        pose_abs = np.hstack((xyz, ang)).astype(np.float32)

        rel = np.zeros_like(pose_abs, dtype=np.float32)
        if self.N > 1:
            rel[1:, :3] = pose_abs[1:, :3] - pose_abs[:-1, :3]
            d = (pose_abs[1:, 3:] - pose_abs[:-1, 3:] + np.pi) % (2 * np.pi) - np.pi
            rel[1:, 3:] = d

        if self.first_mode == "global":
            rel[0] = pose_abs[0]
        elif self.first_mode == "extrapolate" and self.N > 1:
            rel[0, :3] = pose_abs[1, :3] - pose_abs[0, :3]
            ang0 = (pose_abs[1, 3:] - pose_abs[0, 3:] + np.pi) % (2 * np.pi) - np.pi
            rel[0, 3:] = ang0
        # else zero mode → leave rel[0] = 0

        return imgs, imu_batch, torch.from_numpy(rel), self.N
