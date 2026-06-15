import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 網頁基礎設定
st.set_page_config(page_title="偏振光學應力刮痕感測系統", layout="centered", page_icon="🔍")
st.title("🔍 偏振光學應力刮痕感測系統")
st.write("上傳壓克力偏振光刮痕影像，演算法將自動抹平畫素雜訊，並精確推算受力公斤數。")
st.markdown("---")

# 1. 建立網頁上傳按鈕
st.markdown("### 📥 步驟 1：上傳偏振光刮痕照片")
uploaded_file = st.file_uploader("選擇您拍攝的 JPEG/PNG 影像檔案...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 2. 讀取影像並轉為灰階
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # 3. 高斯模糊 (物理抹平 iPad 螢幕畫素網格雜訊)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # 4. 自動裁切與特徵提取 (自動抓取影像中央 200 像素寬度的區域進行垂直平均)
    h, w = blurred.shape
    center_y = int(h / 2)
    y_start, y_end = max(0, center_y - 100), min(h, center_y + 100)
    # ✅ 修改後的新寫法 (限制水平寬度，自動 Crop 刮痕局部)
    center_x = int(w / 2)
    x_start, x_end = max(0, center_x - 150), min(w, center_x + 150)
    crop_area = blurred[y_start:y_end, x_start:x_end]
    
    # 模擬 ImageJ 的 Line Width 平均：對垂直軸做數學平均，消除水平網格
    profile_data = np.mean(crop_area, axis=0)
    
    # 5. 數據演算法：進行移動平均平滑，找出深谷與兩側波峰
    smooth_profile = np.convolve(profile_data, np.ones(11)/11, mode='same')
    
    # 尋找核心特徵值 (抓圖形中央區段，排除邊緣陰影干擾)
    # ✅ 新的寫法：[30:-30] 代表自動切掉左右各 30 像素的斷崖，只抓中央精華訊號！
    safe_zone = smooth_profile[30:-30]
    valley_val = np.min(safe_zone)                     # 抓出精華區真正的刮痕谷底 (約182)
    peak_val = np.max(safe_zone)                       # 抓出精華區真正的應力高峰 (約194)
    delta_H = max(0, peak_val - valley_val)
    
    # 6. 【精準定標實驗公式】結合 1kg(3.8)、3kg(8.5)、5kg(14.0) 三點實驗公式
    if delta_H > 0:
        estimated_force = (delta_H * 0.39) - 0.42
        # 邊界安全防護
        estimated_force = max(0.1, min(estimated_force, 6.5))
    else:
        estimated_force = 0.0
    
    # 7. 介面優化與圖表渲染
    st.markdown("### 📊 步驟 2：演算法演算與數據剖面")
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="原始上傳影像", use_container_width=True)
    with col2:
        # 使用 Matplotlib 繪製平滑後的應力特徵曲線
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(profile_data, color='#CCCCCC', alpha=0.6, label='Raw Data')
        ax.plot(smooth_profile, color='#1E88E5', linewidth=2.5, label='Smoothed')
        ax.set_title("Stress Profile Analysis (ImageJ Auto)")
        ax.set_xlabel("Distance (pixels)")
        ax.set_ylabel("Gray Value")
        ax.grid(True, linestyle='--', alpha=0.5)
        # ✅ 新增這兩行：動態鎖定 Y 軸在數據的上下限，自動放大訊號！
        y_min, y_max = np.min(smooth_profile[15:-15]), np.max(smooth_profile[15:-15])
        ax.set_ylim(y_min - 5, y_max + 5) # 排除邊緣斷崖干擾，上下留 5 個單位的空隙
        
        ax.set_title("Stress Profile Analysis (ImageJ Auto)")
        st.pyplot(fig)
        
    # 8. 炫酷的科學結論噴出
    st.markdown("---")
    st.markdown("### 🎯 步驟 3：智慧型受力強度鑑定")
    
    # 計算顯示進度條
    progress_percentage = min(1.0, estimated_force / 6.0)
    
    if estimated_force >= 4.0:
        st.error(f"🚨 **破壞層級：【 重 度 受 力 】**")
        st.metric(label="推估實際施力 (Force)", value=f"{estimated_force:.2f} kgf", delta=f"+{(estimated_force-3.0):.2f} kgf (高於中度)")
        st.progress(progress_percentage)
        st.caption(f"光學特徵值 ΔH = {delta_H:.2f}。材料內部產生深層應力連鎖形變，屬於高危險破壞。")
    elif estimated_force >= 2.0:
        st.warning(f"⚠️ **破壞層級：【 中 度 受 力 】**")
        st.metric(label="推估實際施力 (Force)", value=f"{estimated_force:.2f} kgf", delta=f"{(estimated_force-3.0):.2f} kgf (趨近基準)")
        st.progress(progress_percentage)
        st.caption(f"光學特徵值 ΔH = {delta_H:.2f}。材料表面出現明顯溝槽，應力集中於刮痕兩側邊緣。")
    else:
        st.success(f"✅ **破壞層級：【 輕 度 受 力 】**")
        st.metric(label="推估實際施力 (Force)", value=f"{estimated_force:.2f} kgf")
        st.progress(progress_percentage)
        st.caption(f"光學特徵值 ΔH = {delta_H:.2f}。僅微弱微觀形變，偏振光干涉條紋極淡。")
