import re
import matplotlib.pyplot as plt
import numpy as np

def extract_decoding_times(filename):
    """从日志文件中提取解码时间数据"""
    with open(filename, 'r') as f:
        content = f.read()
    # 使用正则表达式匹配所有解码时间
    times = re.findall(r'Decoding time per iteration: (\d+\.\d+) seconds', content)
    return [float(t) for t in times]

def plot_comparison(filenames, legend_names=None):
    """
    绘制多个日志文件的解码时间比较图
    :param filenames: 日志文件名列表
    :param legend_names: 自定义图例名称列表（可选）
    """
    # 设置专业绘图风格
    plt.figure(figsize=(10, 6), dpi=300)
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False,
        'axes.linewidth': 0.8
    })
    
    # 如果没有提供自定义图例名称，则使用文件名
    if legend_names is None:
        legend_names = filenames
    
    # 定义颜色和线型配置
    plot_config = {
        'HPF (Ours)': {'color': '#d62728', 'linestyle': '-', 'marker': 'o', 'linewidth': 1.5},
        'LFU': {'color': '#1f77b4', 'linestyle': '-', 'marker': '^', 'linewidth': 1.5}
    }
    
    for filename, legend_name in zip(filenames, legend_names):
        times = extract_decoding_times(filename)
        if not times:
            print(f"警告: 文件 {filename} 中没有找到解码时间数据")
            continue
        
        # 生成token数 (假设每个iteration对应100个token)
        token_counts = [100 * (i+1) for i in range(len(times))]
        
        # 根据图例名称获取对应的绘图配置
        config = plot_config.get(legend_name, {})
        plt.plot(token_counts, times, 
                label=legend_name,
                color=config.get('color'),
                linestyle=config.get('linestyle'),
                marker=config.get('marker', 'o'),
                linewidth=config.get('linewidth', 1.5),
                markersize=4)
        
        # 计算并打印平均解码时间
        avg_time = np.mean(times)
        print(f"{filename}: 平均解码时间 = {avg_time:.6f} 秒/iteration")
    
    # 图表装饰
    plt.title('Decoding Time Comparison', fontsize=14, pad=20)
    plt.xlabel('Number of Generated Tokens', fontsize=12, labelpad=10)
    plt.ylabel('Decoding Time per Iteration (seconds)', fontsize=12, labelpad=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 专业图例设置
    legend = plt.legend(
        title='Strategy',
        title_fontsize=12,
        fontsize=11,
        loc='upper left',
        frameon=True,
        framealpha=0.9,
        edgecolor='#333333',
        facecolor='white',
        borderpad=1
    )
    legend.get_frame().set_linewidth(0.8)
    
    # 坐标轴优化
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=10)
    max_tokens = max([len(extract_decoding_times(f)) * 100 for f in filenames if extract_decoding_times(f)])
    ax.set_xticks(np.arange(0, max_tokens + 100, 500))
    plt.xticks(rotation=45, ha='right')
    
    # 自动调整刻度范围
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(max(0, y_min * 0.9), y_max * 1.05)
    plt.tight_layout()
    plt.savefig('decoding_time_comparison_999.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    # 要分析的日志文件列表
    log_files = ['log_clear_count_priority_999.txt', 'log_clear_count_lfu.txt']

    # 自定义图例名称（与log_files顺序一致）
    custom_legends = [
        'HPF (Ours)',
        'LFU'
    ]
    
    # 绘制比较图
    plot_comparison(log_files, legend_names=custom_legends)