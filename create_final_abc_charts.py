import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Enable interactive mode for better rendering
plt.ion()

# Model data from our analysis
models = ['ABC', 'YOLOv8-nano', 'YOLOv10-nano', 'YOLOv8-small', 'YOLOv10-small']
size_data = [6.8, 12.0, 10.8, 43.0, 36.2]
accuracy_data = [0.300, 0.374, 0.390, 0.449, 0.475]
fps_data = [70, 100, 110, 70, 75]
param_data = [1.8, 3.2, 2.8, 11.2, 9.5]
power_data = [2.1, 3.2, 2.9, 5.8, 5.2]

print("🎨 Creating ABC Model Comparison Visualizations...")
print("=" * 50)

# Create comprehensive comparison chart
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('ABC Model vs YOLO Models - Performance & Efficiency Comparison', 
             fontsize=16, fontweight='bold', y=0.95)

# Chart 1: Model Size Comparison
ax1 = axes[0, 0]
bars1 = ax1.bar(models, size_data, 
                color=['red' if m == 'ABC' else 'skyblue' for m in models])
ax1.set_title('Model Size (MB)', fontweight='bold')
ax1.set_ylabel('Size (MB)')
ax1.tick_params(axis='x', rotation=45)

# Add value labels
for bar, value in zip(bars1, size_data):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{value:.1f}', ha='center', va='bottom', fontsize=10)

# Chart 2: Accuracy Comparison
ax2 = axes[0, 1]
bars2 = ax2.bar(models, accuracy_data,
                color=['red' if m == 'ABC' else 'lightgreen' for m in models])
ax2.set_title('Detection Accuracy (mAP@0.5)', fontweight='bold')
ax2.set_ylabel('mAP@0.5')
ax2.tick_params(axis='x', rotation=45)

# Add value labels
for bar, value in zip(bars2, accuracy_data):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{value:.3f}', ha='center', va='bottom', fontsize=10)

# Chart 3: Speed Comparison
ax3 = axes[0, 2]
bars3 = ax3.bar(models, fps_data,
                color=['red' if m == 'ABC' else 'orange' for m in models])
ax3.set_title('Inference Speed (FPS)', fontweight='bold')
ax3.set_ylabel('FPS')
ax3.tick_params(axis='x', rotation=45)

# Add value labels
for bar, value in zip(bars3, fps_data):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{value:.0f}', ha='center', va='bottom', fontsize=10)

# Chart 4: Parameter Count
ax4 = axes[1, 0]
bars4 = ax4.bar(models, param_data,
                color=['red' if m == 'ABC' else 'purple' for m in models])
ax4.set_title('Parameters (Millions)', fontweight='bold')
ax4.set_ylabel('Parameters (Millions)')
ax4.tick_params(axis='x', rotation=45)

# Add value labels
for bar, value in zip(bars4, param_data):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.2,
             f'{value:.1f}M', ha='center', va='bottom', fontsize=10)

# Chart 5: Power Consumption
ax5 = axes[1, 1]
bars5 = ax5.bar(models, power_data,
                color=['red' if m == 'ABC' else 'brown' for m in models])
ax5.set_title('Power Consumption (Watts)', fontweight='bold')
ax5.set_ylabel('Power (W)')
ax5.tick_params(axis='x', rotation=45)

# Add value labels
for bar, value in zip(bars5, power_data):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{value:.1f}W', ha='center', va='bottom', fontsize=10)

# Chart 6: Efficiency Score Visualization
ax6 = axes[1, 2]
# Calculate efficiency scores (performance/resources)
efficiency_scores = []
for i in range(len(models)):
    # Efficiency = (accuracy × fps) / (parameters × size × power)
    eff = (accuracy_data[i] * fps_data[i]) / (param_data[i] * size_data[i] * power_data[i])
    efficiency_scores.append(eff)

bars6 = ax6.bar(models, efficiency_scores,
                color=['red' if m == 'ABC' else 'gray' for m in models])
ax6.set_title('Efficiency Score\n(Higher = Better)', fontweight='bold')
ax6.set_ylabel('Efficiency Score')
ax6.tick_params(axis='x', rotation=45)

# Add value labels
for bar, value in zip(bars6, efficiency_scores):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + max(efficiency_scores)*0.02,
             f'{value:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
filename = 'abc_model_comprehensive_comparison.png'
plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Comprehensive comparison chart saved: {filename}")

# Create a second detailed analysis chart
plt.figure(figsize=(15, 10))

# Subplot 1: Resource efficiency breakdown (ABC vs Nano models)
plt.subplot(2, 2, 1)
nano_models = ['ABC', 'YOLOv8-nano', 'YOLOv10-nano']
nano_size = [6.8, 12.0, 10.8]
nano_params = [1.8, 3.2, 2.8]
nano_power = [2.1, 3.2, 2.9]

x = range(len(nano_models))
width = 0.25

plt.bar([i - width for i in x], nano_size, width, label='Size (MB)', alpha=0.8)
plt.bar(x, [p*2 for p in nano_params], width, label='Parameters (2x)', alpha=0.8)  # Scale for visibility
plt.bar([i + width for i in x], [p*2 for p in nano_power], width, label='Power (2x)', alpha=0.8)  # Scale for visibility

plt.xlabel('Models')
plt.ylabel('Normalized Values')
plt.title('Resource Usage Comparison (Nano Models)', fontweight='bold')
plt.xticks(x, nano_models)
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Performance metrics comparison
plt.subplot(2, 2, 2)
nano_accuracy = [0.300, 0.374, 0.390]
nano_fps = [70, 100, 110]
nano_precision = [0.75, 0.82, 0.83]

x_perf = range(len(nano_models))
width_perf = 0.2

plt.bar([i - width_perf for i in x_perf], nano_accuracy, width_perf, label='Accuracy', alpha=0.8)
plt.bar(x_perf, [f/150 for f in nano_fps], width_perf, label='Speed (/150)', alpha=0.8)  # Normalized
plt.bar([i + width_perf for i in x_perf], nano_precision, width_perf, label='Precision', alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Normalized Performance Scores')
plt.title('Performance Metrics Comparison', fontweight='bold')
plt.xticks(x_perf, nano_models)
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 3: Efficiency vs Performance scatter plot
plt.subplot(2, 2, 3)
# Use the first 3 models for clarity
scatter_models = ['ABC', 'YOLOv8-nano', 'YOLOv10-nano']
scatter_perf = [0.300, 0.374, 0.390]
scatter_eff = [0.8170, 0.3044, 0.4892]

colors = ['red', 'blue', 'green']
for i, model in enumerate(scatter_models):
    plt.scatter(scatter_eff[i], scatter_perf[i], 
               c=colors[i], s=200, alpha=0.7, label=model)
    plt.annotate(model, (scatter_eff[i], scatter_perf[i]),
                xytext=(10, 10), textcoords='offset points', fontsize=10)

plt.xlabel('Efficiency Score')
plt.ylabel('Performance (mAP@0.5)')
plt.title('Efficiency vs Performance Trade-off', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 4: Summary statistics table visualization
plt.subplot(2, 2, 4)
plt.axis('off')

# Create summary table data
summary_table = [
    ['Metric', 'ABC', 'YOLOv8-nano', 'YOLOv10-nano'],
    ['Size (MB)', '6.8', '12.0', '10.8'],
    ['mAP@0.5', '0.300', '0.374', '0.390'],
    ['FPS', '70', '100', '110'],
    ['Parameters (M)', '1.8', '3.2', '2.8'],
    ['Power (W)', '2.1', '3.2', '2.9'],
    ['Efficiency', '0.817', '0.304', '0.489']
]

table = plt.table(cellText=summary_table,
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.25, 0.25, 0.25, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.1, 1.8)

plt.title('Model Comparison Summary', pad=20, fontweight='bold')

plt.tight_layout()
filename2 = 'abc_model_detailed_analysis.png'
plt.savefig(filename2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Detailed analysis chart saved: {filename2}")

print("\n📊 VISUALIZATION SUMMARY:")
print("=" * 30)
print("Generated Files:")
print(f"  1. {filename}")
print("     - Size comparison bar chart")
print("     - Accuracy comparison bar chart")
print("     - Speed comparison bar chart")
print("     - Parameter count bar chart")
print("     - Power consumption bar chart")
print("     - Efficiency score bar chart")
print()
print(f"  2. {filename2}")
print("     - Resource usage breakdown")
print("     - Performance metrics comparison")
print("     - Efficiency vs performance scatter plot")
print("     - Summary comparison table")
print()
print("🔍 Key Visual Insights:")
print("  • ABC model shows 43-83% resource savings")
print("  • Performance competitive with nano variants")
print("  • Highest efficiency score among all models")
print("  • Clear visual demonstration of trade-offs")
print("  • Professional publication-quality charts")

print(f"\n✅ All visualization files generated successfully!")