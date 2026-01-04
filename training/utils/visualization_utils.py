"""
可视化工具模块
提供基础的绘图功能和通用可视化方法
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import logging
from typing import Optional, Tuple, Any, Dict


class VisualizationUtils:
    """基础可视化工具类"""

    @staticmethod
    def plot_parameter_heatmap(ax, param_img: np.ndarray, title: str, cmap: str = "viridis") -> None:
        """绘制参数热图"""
        im = ax.imshow(param_img, cmap=cmap, interpolation="nearest")
        # 显示统计信息：mean, std, range
        mean_val = param_img.mean()
        std_val = param_img.std()
        min_val = param_img.min()
        max_val = param_img.max()
        ax.set_title(f"{title}\nμ={mean_val:.3f}, σ={std_val:.3f}\n[{min_val:.3f}, {max_val:.3f}]", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    @staticmethod
    def plot_original_image(ax, original_img: Optional[np.ndarray], prompt_info: Optional[Dict] = None) -> None:
        """绘制原始图像，可选择叠加prompt信息"""
        if original_img is not None:
            ax.imshow(original_img)
            
            # 如果有prompt信息，在图像上叠加显示
            if prompt_info is not None:
                VisualizationUtils._overlay_prompts_on_image(ax, prompt_info, original_img.shape)
            
            ax.set_title("Original Image")
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, "No Image\nAvailable", ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title("Original Image")
            ax.axis("off")
    
    @staticmethod
    def _overlay_prompts_on_image(ax, prompt_info: Dict, img_shape: Tuple) -> None:
        """在图像上叠加显示prompt信息
        
        Args:
            ax: matplotlib axis对象
            prompt_info: 包含point_coords和point_labels的字典
            img_shape: 图像shape (H, W, C)
        """
        import torch
        import logging
        
        # 提取点坐标和标签
        point_coords = prompt_info.get("point_coords", None)
        point_labels = prompt_info.get("point_labels", None)
        
        if point_coords is None or point_labels is None:
            return
        
        # 将tensor转换为numpy
        if isinstance(point_coords, torch.Tensor):
            point_coords = point_coords.detach().cpu().numpy()
        if isinstance(point_labels, torch.Tensor):
            point_labels = point_labels.detach().cpu().numpy()
        
        # 通常point_coords的shape是[B, P, 2]，我们只取第一个batch
        if len(point_coords.shape) == 3:
            point_coords = point_coords[0]  # [P, 2]
        if len(point_labels.shape) == 2:
            point_labels = point_labels[0]  # [P]
        
        # SAM2坐标缩放：point_coords是在SAM内部坐标系（通常1024x1024）
        # 需要缩放到实际图像尺寸
        img_h, img_w = img_shape[:2]
        
        # 检测实际的坐标范围
        max_coord = max(point_coords[:, 0].max(), point_coords[:, 1].max())
        
        # 如果坐标值明显大于图像尺寸，说明需要缩放
        if max_coord > max(img_h, img_w) * 1.5:
            # 假设SAM内部使用1024作为基准
            sam_size = 1024.0
            scale_x = img_w / sam_size
            scale_y = img_h / sam_size
            point_coords_scaled = point_coords.copy()
            point_coords_scaled[:, 0] *= scale_x  # x坐标
            point_coords_scaled[:, 1] *= scale_y  # y坐标
        else:
            point_coords_scaled = point_coords
        
        # 统计不同类型的点数量
        num_pos = sum(1 for l in point_labels if l == 1)
        num_neg = sum(1 for l in point_labels if l == 0)
        num_box = sum(1 for l in point_labels if l in [2, 3])
        
        # 绘制每个点
        for i, (coord, label) in enumerate(zip(point_coords_scaled, point_labels)):
            x, y = coord
            
            # 跳过padding点（label=-1）
            if label == -1:
                continue
            
            # 根据label类型选择颜色和标记
            if label == 0:
                color, marker, markersize = 'red', 'x', 12
            elif label == 1:
                color, marker, markersize = 'lime', '*', 15
            elif label in [2, 3]:
                color, marker, markersize = 'cyan', 's', 10
            else:
                color, marker, markersize = 'yellow', 'o', 10
            
            # 绘制点
            ax.plot(x, y, marker=marker, color=color, markersize=markersize,
                   markeredgewidth=2, markeredgecolor='white')
        
        # 如果有box点（label=2和3），绘制矩形框
        box_points = [coord for coord, label in zip(point_coords_scaled, point_labels) if label in [2, 3]]
        
        if len(box_points) == 2:
            x1, y1 = box_points[0]
            x2, y2 = box_points[1]
            from matplotlib.patches import Rectangle
            rect = Rectangle((min(x1, x2), min(y1, y2)), 
                           abs(x2 - x1), abs(y2 - y1),
                           linewidth=2, edgecolor='cyan', facecolor='none',
                           linestyle='--')
            ax.add_patch(rect)
        
        # 记录绘制的prompt类型
        prompt_types = []
        if num_pos > 0:
            prompt_types.append(f"{num_pos} pos")
        if num_neg > 0:
            prompt_types.append(f"{num_neg} neg")
        if num_box > 0:
            prompt_types.append(f"1 box")
        
        if prompt_types:
            logging.info(f"Drew prompts on image: {', '.join(prompt_types)}")

    @staticmethod
    def normalize_parameters_robust(lambda_img: np.ndarray, k_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """稳健的参数归一化，处理异常值"""
        try:
            # 使用百分位数进行稳健归一化，避免异常值影响
            lambda_min = np.percentile(lambda_img, 1)
            lambda_max = np.percentile(lambda_img, 99)
            lambda_range = lambda_max - lambda_min
            if lambda_range < 1e-6:
                lambda_range = 1e-6

            k_min = np.percentile(k_img, 1)
            k_max = np.percentile(k_img, 99)
            k_range = k_max - k_min
            if k_range < 1e-6:
                k_range = 1e-6

            lambda_norm = (lambda_img - lambda_min) / lambda_range
            k_norm = (k_img - k_min) / k_range

            # 限制在[0, 1]范围内
            lambda_norm = np.clip(lambda_norm, 0, 1)
            k_norm = np.clip(k_norm, 0, 1)

            return lambda_norm, k_norm

        except Exception as e:
            logging.warning(f"Parameter normalization failed: {e}")
            # 返回原始值作为fallback
            return lambda_img, k_img

    @staticmethod
    def plot_parameter_overlays(axes, original_img: np.ndarray, lambda_img: np.ndarray, k_img: np.ndarray, step_index: int) -> None:
        """绘制参数叠加图"""
        lambda_norm, k_norm = VisualizationUtils.normalize_parameters_robust(lambda_img, k_img)

        # Lambda叠加
        axes[0].imshow(original_img)
        axes[0].imshow(lambda_norm, cmap="viridis", alpha=0.6, interpolation="nearest")
        axes[0].set_title(f"Lambda Overlay (Step {step_index})")
        axes[0].axis("off")

        # K叠加
        axes[1].imshow(original_img)
        axes[1].imshow(k_norm, cmap="plasma", alpha=0.6, interpolation="nearest")
        axes[1].set_title(f"K Overlay (Step {step_index})")
        axes[1].axis("off")

        # 组合叠加
        axes[2].imshow(original_img)
        combined = np.zeros((*lambda_img.shape, 3))
        combined[:, :, 1] = lambda_norm  # Green for lambda
        combined[:, :, 0] = k_norm  # Red for k
        axes[2].imshow(combined, alpha=0.6, interpolation="nearest")
        axes[2].set_title(f"Combined Overlay (Step {step_index})")
        axes[2].axis("off")

    @staticmethod
    def plot_parameter_distributions(axes, lambda_img: np.ndarray, k_img: np.ndarray, step_index: int) -> None:
        """绘制参数分布图"""
        axes[0].hist(lambda_img.flatten(), bins=50, alpha=0.7, color="green")
        axes[0].set_title(f"Lambda Distribution (Step {step_index})\nMean: {lambda_img.mean():.4f}")

        axes[1].hist(k_img.flatten(), bins=50, alpha=0.7, color="red")
        axes[1].set_title(f"K Distribution (Step {step_index})\nMean: {k_img.mean():.4f}")

        param_diff = lambda_img - k_img
        im = axes[2].imshow(param_diff, cmap="RdBu", interpolation="nearest")
        axes[2].set_title(f"Lambda - K Difference (Step {step_index})")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    @staticmethod
    def plot_metric_uncertainty_correlation(ax, uncertainty: np.ndarray, metric: np.ndarray, title: str, xlabel: str, ylabel: str, step_index: int) -> None:
        """绘制指标与不确定性的相关性图"""
        try:
            # 子采样以提高性能
            if len(uncertainty) > 10000:
                indices = np.random.choice(len(uncertainty), 10000, replace=False)
                uncertainty = uncertainty[indices]
                metric = metric[indices]

            # 创建散点图
            ax.scatter(uncertainty, metric, alpha=0.6, s=1, c=metric, cmap="viridis")

            # 计算并绘制相关系数
            if len(uncertainty) > 1 and len(metric) > 1:
                correlation = np.corrcoef(uncertainty, metric)[0, 1]
                if not np.isnan(correlation):
                    ax.text(0.05, 0.95, f"Corr: {correlation:.3f}", transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            # 添加趋势线
            try:
                z = np.polyfit(uncertainty, metric, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(uncertainty.min(), uncertainty.max(), 100)
                ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=1)
            except:
                pass

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} (Step {step_index})")
            ax.grid(True, alpha=0.3)

            # 添加基本统计信息
            stats_text = f"Mean: {metric.mean():.3f}\nStd: {metric.std():.3f}"
            ax.text(0.05, 0.05, stats_text, transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=8)

        except Exception as e:
            logging.warning(f"Failed to plot {title}: {e}")
            ax.text(0.5, 0.5, f"{title}\nPlot Failed", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{title} (Error)")

    @staticmethod
    def plot_dataset_metric_uncertainty_correlation(fig: plt.Figure, correlation_results: Dict[str, Dict[str, float]], title: str = "Dataset Metric-Uncertainty Correlation Analysis") -> None:
        """
        绘制整个数据集的指标与不确定性相关性分析图 - 散点图版本
        
        Args:
            fig: matplotlib图形对象
            correlation_results: 从MetricCalculator.calculate_dataset_metric_uncertainty_correlation返回的结果
            title: 图表标题
        """
        try:
            if not correlation_results:
                logging.warning("No correlation results to plot")
                return

            # 清除现有内容
            fig.clear()

            # 计算子图布局
            num_metrics = len(correlation_results)
            if num_metrics <= 3:
                cols = num_metrics
                rows = 1
            else:
                cols = 3
                rows = (num_metrics + cols - 1) // cols

            # 创建子图，确保有足够的空间
            gs = fig.add_gridspec(rows + 1, cols, hspace=0.5, wspace=0.4, height_ratios=[1] * rows + [0.35])

            # 设置总标题
            fig.suptitle(title, fontsize=16, fontweight="bold")

            # 为每个指标创建散点图子图
            for i, (metric_name, results) in enumerate(correlation_results.items()):
                row = i // cols
                col = i % cols

                ax = fig.add_subplot(gs[row, col])

                # 检查是否有原始数据用于绘制散点图
                if 'uncertainty_data' in results and 'metric_data' in results:
                    # 绘制散点图
                    uncertainty_data = results['uncertainty_data']
                    metric_data = results['metric_data']
                    
                    # 确保数据是numpy数组
                    if isinstance(uncertainty_data, list):
                        uncertainty_data = np.array(uncertainty_data)
                    if isinstance(metric_data, list):
                        metric_data = np.array(metric_data)
                    
                    # 使用hexbin替代传统散点，缓解重叠；并叠加分位均值曲线
                    hb = ax.hexbin(uncertainty_data, metric_data, gridsize=40, cmap='viridis', mincnt=5)
                    cb = fig.colorbar(hb, ax=ax)
                    cb.set_label('Counts')

                    # 分位分箱UA曲线
                    try:
                        q = np.linspace(0.0, 1.0, 21)
                        bins = np.quantile(uncertainty_data, q)
                        bins[0] = np.min(uncertainty_data)
                        bins[-1] = np.max(uncertainty_data)
                        inds = np.digitize(uncertainty_data, bins, right=True)
                        x_centers = []
                        y_means = []
                        for b in range(1, len(bins)):
                            mask = inds == b
                            if np.sum(mask) >= 10:
                                x_centers.append(np.median(uncertainty_data[mask]))
                                y_means.append(np.mean(metric_data[mask]))
                        if x_centers:
                            ax.plot(x_centers, y_means, color='orange', linewidth=2, label='UA-curve (quantile means)')
                            ax.legend()
                    except Exception:
                        pass
                    
                    # 绘制趋势线
                    if 'slope' in results and 'intercept' in results and not np.isnan(results['slope']):
                        slope = results['slope']
                        intercept = results['intercept']
                        x_min, x_max = uncertainty_data.min(), uncertainty_data.max()
                        y_trend = slope * np.array([x_min, x_max]) + intercept
                        ax.plot([x_min, x_max], y_trend, 'r-', linewidth=2, label=f'Slope: {slope:.4f}')
                        ax.legend()
                    
                    # 设置轴标签
                    ax.set_xlabel('Uncertainty (higher = less certain)', fontsize=10)
                    ax.set_ylabel(metric_name, fontsize=10)
                    ax.set_title(f'{metric_name} vs Uncertainty', fontweight='bold', fontsize=12)
                    
                    # 添加网格
                    ax.grid(True, alpha=0.3)
                    
                    # 在图上显示关键统计信息
                    correlation = results.get('correlation', np.nan)
                    if not np.isnan(correlation):
                        ax.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
                               transform=ax.transAxes, fontsize=10, 
                               bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                               verticalalignment='top')
                    
                else:
                    # 如果没有原始数据，显示统计信息
                    stats_text = f"""
                        {metric_name}
                        
                        Correlation: {results.get("correlation", "N/A"):.4f}
                        Slope: {results.get("slope", "N/A"):.4f}
                        Valid Points: {results.get("num_valid_points", "N/A")}/{results.get("total_points", "N/A")}
                    """.strip()
                    
                    ax.text(0.5, 0.5, stats_text, transform=ax.transAxes, 
                           ha="center", va="center", fontsize=10, 
                           bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))
                    ax.set_title(f"{metric_name}", fontweight="bold")
                    ax.axis("off")

            # 添加汇总统计表格
            if num_metrics > 0:
                # 在底部添加汇总表格
                summary_ax = fig.add_subplot(gs[-1, :])
                summary_ax.axis("off")

                # 创建汇总表格
                summary_data = []
                for metric_name, results in correlation_results.items():
                    correlation = results.get('correlation', 'N/A')
                    slope = results.get('slope', 'N/A')
                    valid_points = results.get('num_valid_points', 'N/A')
                    total_points = results.get('total_points', 'N/A')
                    
                    summary_data.append([
                        metric_name,
                        f"{correlation:.4f}" if isinstance(correlation, (int, float)) else str(correlation),
                        f"{slope:.4f}" if isinstance(slope, (int, float)) else str(slope),
                        f"{valid_points}/{total_points}",
                    ])

                table = summary_ax.table(cellText=summary_data, 
                                       colLabels=["Metric", "Correlation", "Slope", "Valid Points"], 
                                       cellLoc="center", loc="center")
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 2)

                summary_ax.set_title("Summary Statistics", fontweight="bold", pad=20)

        except Exception as e:
            logging.warning(f"Failed to plot dataset correlation analysis: {e}")
            import traceback
            logging.warning(f"Traceback: {traceback.format_exc()}")

    @staticmethod
    def create_figure_layout(rows: int, cols: int = 3, figsize: Tuple[int, int] = (18, 6)) -> Tuple[plt.Figure, np.ndarray]:
        """创建图表布局"""
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        return fig, axes

    @staticmethod
    def save_and_close_figure(fig: plt.Figure, save_path: str, dpi: int = 150, close_fig: bool = True, save_pdf: bool = False) -> None:
        """保存并关闭图表
        
        Args:
            fig: Matplotlib figure to save
            save_path: Path to save the figure (can be .png or .pdf)
            dpi: DPI resolution for saving (default: 150, use 300 for paper-quality)
            close_fig: Whether to close the figure after saving (default: True)
            save_pdf: If True, save both PNG and PDF versions (PDF at 300 DPI for paper)
        """
        import os
        plt.figure(fig.number)
        plt.tight_layout()
        
        # Determine output format based on extension
        base_path, ext = os.path.splitext(save_path)
        
        if ext.lower() == '.pdf':
            # If explicitly PDF, save as PDF only with 300 DPI
            plt.savefig(save_path, dpi=300, format='pdf', bbox_inches='tight')
        else:
            # Save PNG
            plt.savefig(save_path, dpi=dpi)
            
            # Optionally also save PDF for paper figures
            if save_pdf:
                pdf_path = base_path + '.pdf'
                plt.savefig(pdf_path, dpi=300, format='pdf', bbox_inches='tight')
        
        if close_fig:
            plt.close(fig)
    
    @staticmethod
    def create_ua_ratio_visualization(
        out_logits: np.ndarray,
        uncertainty_map: np.ndarray,
        original_img: np.ndarray,
        vid: str,
        frame_name: str,
        vis_dir: Any,
        method_name: str = "Method",
    ) -> None:
        """创建通用的U/A ratio可视化
        
        适用于所有方法（UCTTA, UR-ERN, BNDL等）的统一可视化接口
        
        Args:
            out_logits: 预测logits [1, H, W, K] 或其他格式
            uncertainty_map: 不确定性图 [1, H, W] 或 [H, W]
            original_img: 原始图像 [H, W, 3]
            vid: 视频名称
            frame_name: 帧名称
            vis_dir: 可视化输出目录（Path对象）
            method_name: 方法名称（用于标题）
        """
        try:
            from pathlib import Path
            from bndl_visualizer import BNDLVisualizer
            
            # 确保vis_dir是Path对象
            if not isinstance(vis_dir, Path):
                vis_dir = Path(vis_dir)
            
            viz_utils = VisualizationUtils()
            bndl_viz = BNDLVisualizer()
            
            # 准备数据格式（与BNDL visualizer期望的格式一致）
            method_outputs = {
                "pixel_uncertainty": uncertainty_map,
                "mean_pixel_logits": out_logits,
            }
            
            # 创建figure
            fig, axes = viz_utils.create_figure_layout(1, 3, (18, 6))
            
            # 使用BNDL visualizer的U/A ratio可视化
            bndl_viz.plot_uncertainty_accuracy_ratio_visualization(
                axes[0, :], method_outputs, original_img, step_index=0, ratio_type="U/A"
            )
            
            # 更新标题以包含方法名
            if axes[0, 0].get_title():
                axes[0, 0].set_title(f"{method_name}: {axes[0, 0].get_title()}")
            
            # 保存可视化
            save_path = vis_dir / vid / f"{frame_name}_{method_name.lower().replace(' ', '_')}_ua_ratio.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            viz_utils.save_and_close_figure(fig, str(save_path), dpi=150)
            
        except Exception as e:
            logging.warning(f"Failed to create UA ratio visualization for {method_name}: {e}")
            import traceback
            logging.warning(f"Traceback: {traceback.format_exc()}")
    
    @staticmethod
    def get_component_names(rows: int, has_uncertainty: bool, has_pavpu: bool, has_ratio_data: bool) -> Dict[Tuple[int, int], str]:
        """Generate descriptive names for each subplot component.
        
        Args:
            rows: Number of rows in the visualization
            has_uncertainty: Whether uncertainty visualization is present
            has_pavpu: Whether PAvPU overlay visualization is present
            has_ratio_data: Whether U/A ratio visualization is present
        
        Returns:
            Dict mapping (row, col) tuples to descriptive names
        """
        names = {
            # Row 0: Basic parameters
            (0, 0): "original_image",
            (0, 1): "lambda_heatmap",
            (0, 2): "k_heatmap",
            # Row 1: Overlays
            (1, 0): "lambda_overlay",
            (1, 1): "k_overlay",
            (1, 2): "combined_overlay",
            # Row 2: Global parameters (spans 3 columns)
            (2, 0): "global_params_left",
            (2, 1): "global_params_center",
            (2, 2): "global_params_right",
        }
        
        current_row = 3
        if has_uncertainty and rows >= 4:
            names[(current_row, 0)] = "uncertainty_left"
            names[(current_row, 1)] = "uncertainty_center"
            names[(current_row, 2)] = "uncertainty_right"
            current_row += 1
        
        if has_pavpu and rows >= current_row + 1:
            names[(current_row, 0)] = "pavpu_overlay_left"
            names[(current_row, 1)] = "pavpu_overlay_center"
            names[(current_row, 2)] = "pavpu_overlay_right"
            current_row += 1
        
        if has_ratio_data and rows >= current_row + 1:
            names[(current_row, 0)] = "ua_ratio_left"
            names[(current_row, 1)] = "ua_ratio_center"
            names[(current_row, 2)] = "ua_ratio_right"
        
        return names
    
    @staticmethod
    def save_individual_subplots(fig: plt.Figure, axes: np.ndarray, save_dir: str, component_names: Dict[Tuple[int, int], str]) -> None:
        """Save each subplot as an individual image file.
        
        Args:
            fig: Matplotlib figure object
            axes: Array of subplot axes
            save_dir: Directory to save individual components
            component_names: Dict mapping (row, col) to descriptive names
        """
        try:
            # Flatten axes array for easier iteration
            if axes.ndim == 1:
                axes_flat = axes
                rows, cols = len(axes), 1
            else:
                axes_flat = axes.flatten()
                rows, cols = axes.shape
            
            # Save each subplot individually
            for idx, ax in enumerate(axes_flat):
                row = idx // cols
                col = idx % cols
                
                # Get component name
                component_name = component_names.get((row, col), f"component_{row}_{col}")
                
                # Skip empty subplots
                if not ax.has_data() and not ax.get_title() and not ax.texts:
                    continue
                
                # Create a new figure for this subplot
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                
                # Save with some padding
                import os
                save_path = os.path.join(save_dir, f"{row:02d}_{col:02d}_{component_name}.png")
                fig.savefig(save_path, bbox_inches=extent.expanded(1.1, 1.1), dpi=150)
            
            logging.info(f"Saved {len(component_names)} individual subplot components")
        except Exception as e:
            logging.warning(f"Failed to save individual subplots: {e}")