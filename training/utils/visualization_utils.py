"""
可视化工具模块
提供基础的绘图功能和通用可视化方法
"""

import numpy as np
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
        ax.set_title(f"{title}\nMean: {param_img.mean():.4f}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    @staticmethod
    def plot_original_image(ax, original_img: Optional[np.ndarray]) -> None:
        """绘制原始图像"""
        if original_img is not None:
            ax.imshow(original_img)
            ax.set_title("Original Image")
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, "No Image\nAvailable", ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title("Original Image")
            ax.axis("off")

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
            gs = fig.add_gridspec(rows + 1, cols, hspace=0.4, wspace=0.3, height_ratios=[1] * rows + [0.3])

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
                    
                    # 创建散点图，使用透明度避免过度重叠
                    ax.scatter(uncertainty_data, metric_data, alpha=0.6, s=1, color='blue')
                    
                    # 绘制趋势线
                    if 'slope' in results and 'intercept' in results and not np.isnan(results['slope']):
                        slope = results['slope']
                        intercept = results['intercept']
                        x_min, x_max = uncertainty_data.min(), uncertainty_data.max()
                        y_trend = slope * np.array([x_min, x_max]) + intercept
                        ax.plot([x_min, x_max], y_trend, 'r-', linewidth=2, label=f'Slope: {slope:.4f}')
                        ax.legend()
                    
                    # 设置轴标签
                    ax.set_xlabel('Uncertainty', fontsize=10)
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
    def save_and_close_figure(fig: plt.Figure, save_path: str, dpi: int = 150) -> None:
        """保存并关闭图表"""
        plt.figure(fig.number)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
        plt.close(fig)
