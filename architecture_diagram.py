"""
Zero-DCE++ Architecture Diagram Generator
Creates a comprehensive visualization of the DCENet with SE-Block model architecture
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_architecture_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Title
    ax.text(10, 13.5, 'Zero-DCE++ Architecture with SE-Block', 
            fontsize=18, fontweight='bold', ha='center', va='center')
    ax.text(10, 12.9, 'DCENet for Low-Light Image Enhancement', 
            fontsize=12, ha='center', va='center', style='italic', color='gray')
    
    # Color scheme
    colors = {
        'input': '#E3F2FD',      # Light blue
        'conv': '#C8E6C9',       # Light green
        'se': '#FFE0B2',         # Light orange
        'relu': '#E1F5FE',       # Very light blue
        'concat': '#F3E5F5',     # Light purple
        'tanh': '#FFF9C4',       # Light yellow
        'output': '#FFCDD2',     # Light red
        'curve': '#E0F7FA',      # Cyan tint
        'arrow': '#424242'       # Dark gray
    }
    
    def draw_box(x, y, width, height, text, color, fontsize=9, bold=False):
        box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                              boxstyle="round,pad=0.05,rounding_size=0.2",
                              facecolor=color, edgecolor='#333', linewidth=1.5)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x, y, text, fontsize=fontsize, ha='center', va='center',
                fontweight=weight, wrap=True)
    
    def draw_arrow(start, end, color='#424242'):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    # ============== INPUT SECTION ==============
    draw_box(2, 8, 2.5, 1.5, 'Input Image\n(3 x H x W)', colors['input'], fontsize=10, bold=True)
    
    # ============== ENCODER PATH ==============
    # conv1
    draw_box(4.5, 8, 2.2, 1.2, 'Conv2d\n3->32, 3x3', colors['conv'], fontsize=9)
    draw_arrow((3.25, 8), (3.4, 8))
    
    draw_box(4.5, 6.8, 1.5, 0.8, 'ReLU', colors['relu'], fontsize=8)
    draw_arrow((4.5, 7.4), (4.5, 7.2))
    
    # x1 label
    ax.text(3.6, 8.5, 'x1', fontsize=10, fontweight='bold', color='#1976D2')
    
    # conv2
    draw_box(7, 8, 2.2, 1.2, 'Conv2d\n32->32, 3x3', colors['conv'], fontsize=9)
    draw_arrow((5.6, 8), (5.9, 8))
    
    draw_box(7, 6.8, 1.5, 0.8, 'ReLU', colors['relu'], fontsize=8)
    draw_arrow((7, 7.4), (7, 7.2))
    
    # x2 label
    ax.text(6.1, 8.5, 'x2', fontsize=10, fontweight='bold', color='#1976D2')
    
    # conv3
    draw_box(9.5, 8, 2.2, 1.2, 'Conv2d\n32->32, 3x3', colors['conv'], fontsize=9)
    draw_arrow((8.1, 8), (8.4, 8))
    
    draw_box(9.5, 6.8, 1.5, 0.8, 'ReLU', colors['relu'], fontsize=8)
    draw_arrow((9.5, 7.4), (9.5, 7.2))
    
    # x3 label
    ax.text(8.6, 8.5, 'x3', fontsize=10, fontweight='bold', color='#1976D2')
    
    # ============== SE-BLOCK ==============
    # SE Block box
    se_box = FancyBboxPatch((8.2, 4.5), 2.8, 1.8,
                            boxstyle="round,pad=0.1,rounding_size=0.3",
                            facecolor=colors['se'], edgecolor='#E65100', linewidth=2)
    ax.add_patch(se_box)
    ax.text(9.6, 5.9, 'SE-Block', fontsize=10, fontweight='bold', ha='center', color='#E65100')
    
    # SE Block internal structure
    draw_box(9.1, 5.4, 1.3, 0.7, 'Adaptive\nAvgPool', '#FFCCBC', fontsize=7)
    draw_box(10.1, 5.4, 1.3, 0.7, 'FC->r\nFC->c\nSigmoid', '#FFCCBC', fontsize=7)
    
    ax.text(9.1, 5.0, '(c->c/r)', fontsize=7, ha='center', color='gray')
    ax.text(10.1, 5.0, '(c/r->c)', fontsize=7, ha='center', color='gray')
    
    # Arrow to SE
    draw_arrow((9.5, 6.4), (9.1, 5.85))
    
    # Arrow from SE output
    ax.annotate('', xy=(10.5, 5.85), xytext=(10.1, 5.4),
                arrowprops=dict(arrowstyle='->', color='#424242', lw=1.5))
    
    # Multiply symbol
    ax.text(10.8, 5.85, 'x', fontsize=14, fontweight='bold', ha='center')
    ax.text(11.1, 5.85, 'x3', fontsize=9, ha='left', style='italic')
    
    # ============== DECODER PATH ==============
    # conv4
    draw_box(12, 8, 2.2, 1.2, 'Conv2d\n32->32, 3x3', colors['conv'], fontsize=9)
    draw_arrow((11.5, 5.85), (12, 8), colors['arrow'])
    
    draw_box(12, 6.8, 1.5, 0.8, 'ReLU', colors['relu'], fontsize=8)
    draw_arrow((12, 7.4), (12, 7.2))
    
    # x4 label
    ax.text(12, 8.5, 'x4', fontsize=10, fontweight='bold', color='#1976D2')
    
    # Concat 1 (x4 + x3)
    draw_box(13.5, 5.85, 1.8, 1.2, 'Concat\n(x4 + x3)\n64 channels', colors['concat'], fontsize=8)
    draw_arrow((12, 6.4), (13.5, 6.45))
    ax.annotate('', xy=(12.6, 5.85), xytext=(12, 5.85),
                arrowprops=dict(arrowstyle='->', color='#424242', lw=1.5))
    
    # conv5
    draw_box(15.2, 5.85, 2.2, 1.2, 'Conv2d\n64->32, 3x3', colors['conv'], fontsize=9)
    draw_arrow((14.4, 5.85), (14.1, 5.85))
    
    draw_box(15.2, 4.65, 1.5, 0.8, 'ReLU', colors['relu'], fontsize=8)
    draw_arrow((15.2, 5.25), (15.2, 5.05))
    
    # x5 label
    ax.text(15.2, 6.5, 'x5', fontsize=10, fontweight='bold', color='#1976D2')
    
    # Concat 2 (x5 + x2)
    draw_box(16.5, 3.45, 1.8, 1.2, 'Concat\n(x5 + x2)\n64 channels', colors['concat'], fontsize=8)
    draw_arrow((15.2, 4.25), (16.5, 4.05))
    ax.annotate('', xy=(15.7, 3.45), xytext=(15.2, 3.45),
                arrowprops=dict(arrowstyle='->', color='#424242', lw=1.5))
    
    # conv6
    draw_box(16.5, 2.2, 2.2, 1.2, 'Conv2d\n64->32, 3x3', colors['conv'], fontsize=9)
    draw_arrow((15.6, 3.45), (15.4, 3.45))
    
    draw_box(16.5, 1.0, 1.5, 0.8, 'ReLU', colors['relu'], fontsize=8)
    draw_arrow((16.5, 1.6), (16.5, 1.4))
    
    # x6 label
    ax.text(16.5, 2.8, 'x6', fontsize=10, fontweight='bold', color='#1976D2')
    
    # ============== OUTPUT LAYER ==============
    # Concat 3 (x6 + x1)
    draw_box(17.5, 5.85, 1.8, 1.2, 'Concat\n(x6 + x1)\n64 channels', colors['concat'], fontsize=8)
    
    # x1 arrow going down
    ax.annotate('', xy=(16.6, 5.85), xytext=(3.6, 8),
                arrowprops=dict(arrowstyle='->', color='#9E9E9E', lw=1.5, 
                               connectionstyle='arc3,rad=-0.2'))
    
    draw_arrow((16.5, 2.2), (17.5, 5.85))
    
    # conv7 (output)
    draw_box(17.5, 8, 2.2, 1.2, 'Conv2d\n64->24, 3x3', colors['conv'], fontsize=9)
    draw_arrow((17.5, 5.85), (17.5, 7.4))
    
    draw_box(17.5, 9.3, 1.5, 0.8, 'Tanh', colors['tanh'], fontsize=9)
    draw_arrow((17.5, 8.6), (17.5, 8.9))
    
    # Output: Alpha (24 channels)
    ax.text(17.5, 10.2, 'Alpha = 24 maps', fontsize=10, fontweight='bold', ha='center')
    ax.text(17.5, 9.8, '(8 curve params x 3 channels)', fontsize=8, ha='center', color='gray')
    
    # ============== CURVE APPLICATION ==============
    # Curve enhancement box
    curve_box = FancyBboxPatch((13, 9.8), 4, 2.2,
                                boxstyle="round,pad=0.1,rounding_size=0.3",
                                facecolor=colors['curve'], edgecolor='#006064', linewidth=2)
    ax.add_patch(curve_box)
    ax.text(15, 11.5, 'Curve Enhancement', fontsize=11, fontweight='bold', ha='center', color='#006064')
    
    # Formula
    ax.text(15, 10.7, 'enhanced = enhanced', fontsize=9, ha='center', family='monospace')
    ax.text(15, 10.35, '+ alpha_i x enhanced x (1 - enhanced)', fontsize=9, ha='center', family='monospace')
    ax.text(15, 10.0, 'for i = 1 to 8 iterations', fontsize=8, ha='center', style='italic', color='gray')
    
    # Arrow from tanh to curve
    ax.annotate('', xy=(15, 9.8), xytext=(17.5, 9.7),
                arrowprops=dict(arrowstyle='->', color='#006064', lw=2))
    
    # ============== FINAL OUTPUT ==============
    draw_box(15, 8, 2.5, 1.5, 'Enhanced\nImage', colors['output'], fontsize=10, bold=True)
    draw_arrow((15, 9.8), (15, 8.75))
    
    # ============== LEGEND ==============
    legend_y = 0.8
    legend_items = [
        (colors['input'], 'Input/Output'),
        (colors['conv'], 'Conv2d Layer'),
        (colors['relu'], 'ReLU Activation'),
        (colors['se'], 'SE-Block'),
        (colors['concat'], 'Concatenation'),
        (colors['tanh'], 'Tanh Activation'),
    ]
    
    ax.text(2, legend_y + 0.5, 'Legend:', fontsize=10, fontweight='bold')
    for i, (color, label) in enumerate(legend_items):
        x = 2 + (i % 3) * 3
        y = legend_y - (i // 3) * 0.5
        rect = plt.Rectangle((x - 0.15, y - 0.15), 0.3, 0.3, facecolor=color, edgecolor='#333')
        ax.add_patch(rect)
        ax.text(x + 0.3, y, label, fontsize=8, va='center')
    
    # ============== MODEL SUMMARY ==============
    summary_text = """Model Summary:
- Total Parameters: ~79K
- Input: 3 x H x W (RGB)
- Output: 24 channel maps
- SE-Block: channel reduction=8
- Skip connections at 3 decoder levels"""
    
    ax.text(10, 0.3, summary_text, fontsize=8, ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
            family='monospace')
    
    plt.tight_layout()
    plt.savefig('zerodce_architecture_diagram.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('zerodce_architecture_diagram.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Architecture diagram saved: zerodce_architecture_diagram.png")
    print("Architecture diagram saved: zerodce_architecture_diagram.pdf")
    plt.show()


def create_detailed_se_block_diagram():
    """Create a detailed SE-Block diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(7, 9.5, 'Squeeze-and-Excitation (SE) Block Detail', 
            fontsize=16, fontweight='bold', ha='center')
    
    def draw_box(x, y, w, h, t, c, fs=10):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                                   boxstyle="round,pad=0.05", facecolor=c, edgecolor='#333'))
        ax.text(x, y, t, fontsize=fs, ha='center', va='center')
    
    # Input feature map
    draw_box(2, 5, 2.5, 2, 'Input\n(CxHxW)\n32xWxH', '#E3F2FD', 10)
    
    # Squeeze - Global Pool
    draw_box(5, 5, 2.5, 1.5, 'Squeeze\nAdaptiveAvgPool2d\n(32,1,1)', '#FFCCBC', 9)
    ax.annotate('', xy=(4.25, 5), xytext=(3.25, 5),
                arrowprops=dict(arrowstyle='->', lw=2))
    
    # FC layers
    draw_box(8, 5, 2.2, 2.2, 'Excitation\nFC(32->4) ReLU\nFC(4->32) Sigmoid', '#FFE0B2', 9)
    ax.annotate('', xy=(6.9, 5), xytext=(6.1, 5),
                arrowprops=dict(arrowstyle='->', lw=2))
    
    # Scale (Multiply)
    draw_box(11, 5, 2, 1.5, 'Scale\nF = sigmoid()\nOutput x F', '#C8E6C9', 9)
    ax.annotate('', xy=(10, 5), xytext=(9.1, 5),
                arrowprops=dict(arrowstyle='->', lw=2))
    
    # Output
    draw_box(13, 5, 1.8, 1.5, 'Output\n(CxHxW)\n32xWxH', '#E8F5E9', 10)
    ax.annotate('', xy=(12.1, 5), xytext=(12, 5),
                arrowprops=dict(arrowstyle='->', lw=2))
    
    # Formula box
    formula = """SE-Block Formula:
1. Squeeze: z_c = (1/(HxW)) Sum z_ij
2. Excitation: s = sigma(W2.ReLU(W1.z))
3. Scale: y_c = F_c x s_c

Where:
- r = reduction ratio (default: 8)
- W1: (C/r x C), W2: (C x C/r)"""
    
    ax.text(7, 1.5, formula, fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#FFF8E1', alpha=0.9, edgecolor='#FFC107'),
            family='monospace')
    
    plt.tight_layout()
    plt.savefig('se_block_detail.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    print("SE-Block detail saved: se_block_detail.png")
    plt.show()


def create_loss_function_diagram():
    """Create a diagram showing the loss functions"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(8, 7.5, 'Zero-DCE++ v4 Loss Functions', 
            fontsize=16, fontweight='bold', ha='center')
    
    colors = {
        'spa': '#4CAF50',
        'exp': '#2196F3', 
        'col': '#9C27B0',
        'tva': '#FF5722'
    }
    
    # Loss components
    losses = [
        ('L_spa', 'Spatial Consistency', 
         'MSE(grad_E_enh, grad_E_low)', colors['spa'], 2.5),
        ('L_exp', 'Exposure Control', 
         'MSE(patches, E=0.6)', colors['exp'], 6),
        ('L_col', 'Color Constancy', 
         '|R-G| + |G-B| + |B-R|', colors['col'], 9.5),
        ('L_tv_a', 'Smoothness', 
         'TV(alpha_x) + TV(alpha_y)', colors['tva'], 13)
    ]
    
    for name, title, formula, color, x in losses:
        # Box
        ax.add_patch(FancyBboxPatch((x-1.3, 2.5), 2.6, 3,
                                    boxstyle="round,pad=0.1", 
                                    facecolor=color, alpha=0.2, edgecolor=color, linewidth=2))
        
        ax.text(x, 5, name, fontsize=14, fontweight='bold', ha='center', color=color)
        ax.text(x, 4.2, title, fontsize=10, ha='center')
        ax.text(x, 3.2, formula, fontsize=8, ha='center', family='monospace',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Total loss formula
    ax.text(8, 1.5, 
            'L_total = L_spa + 20.L_exp + 5.L_col + 1600.L_tv_a',
            fontsize=12, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2),
            family='monospace')
    
    # Weights annotation
    ax.text(8, 0.5, 
            'Note: L_tv_enh (v3) was removed in v4 due to mode collapse issues',
            fontsize=9, ha='center', style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig('loss_functions_diagram.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    print("Loss functions diagram saved: loss_functions_diagram.png")
    plt.show()


if __name__ == "__main__":
    print("Generating Zero-DCE++ Architecture Diagrams...")
    print("=" * 50)
    
    create_architecture_diagram()
    create_detailed_se_block_diagram()
    create_loss_function_diagram()
    
    print("=" * 50)
    print("All diagrams generated successfully!")
