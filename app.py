import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --- CÀI ĐẶT TRANG ---
st.set_page_config(page_title="Hệ thống Đề xuất Sản phẩm", layout="wide")

# --- HẰNG SỐ ---
PRIORITY_THRESHOLD_HIGH = 2.1
PRIORITY_THRESHOLD_LOW = 1.9

# --- ĐƯỜNG DẪN TỆP ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "kmeans_pipeline.joblib"
PROFILES_PATH = BASE_DIR / "cluster_profiles.csv"
ORIGINAL_DATA_PATH = BASE_DIR / "khach_hang.csv"


@st.cache_resource
def load_model_and_profiles(model_path: Path, profiles_path: Path):
    """Tải pipeline và hồ sơ cụm đã được huấn luyện.

    Trả về (pipeline, cluster_profiles_df) hoặc (None, None) nếu không tìm thấy.
    """
    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        return None, None

    try:
        cluster_profiles_df = pd.read_csv(profiles_path, index_col='Cluster')
    except Exception as e:
        st.error(f"Lỗi khi đọc hồ sơ cụm: {e}")
        return pipeline, None

    return pipeline, cluster_profiles_df


pipeline, cluster_profiles = load_model_and_profiles(MODEL_PATH, PROFILES_PATH)


def safe_get_profile_value(profile, key, default=2.0):
    """Lấy giá trị từ profile an toàn, trả về default nếu thiếu hoặc NaN."""
    try:
        val = profile.get(key, default)
        if pd.isna(val):
            return default
        return val
    except Exception:
        return default


def get_product_recommendation(cluster_id, profiles):
    """Tạo đề xuất sản phẩm dựa trên hồ sơ cụm."""
    if profiles is None:
        return "Không có hồ sơ cụm", "Hồ sơ cụm chưa được nạp."

    if str(cluster_id) not in profiles.index.astype(str).tolist():
        return "Không tìm thấy thông tin cho cụm này.", "Vui lòng kiểm tra lại."

    # Hỗ trợ cả index kiểu số và chuỗi
    if cluster_id in profiles.index:
        profile = profiles.loc[cluster_id]
    else:
        profile = profiles.loc[int(cluster_id)] if str(cluster_id).isdigit() else profiles.loc[cluster_id]

    # Lấy các giá trị từ profile một cách an toàn
    ut_thanhphan = float(safe_get_profile_value(profile, 'UT_ThanhPhan', 2.0))
    ut_giaca = float(safe_get_profile_value(profile, 'UT_GiaCa', 2.0))
    ut_congdung = float(safe_get_profile_value(profile, 'UT_CongDung', 2.0))
    ut_muihuong = float(safe_get_profile_value(profile, 'UT_MuiHuong', 2.0))
    ut_baobi = float(safe_get_profile_value(profile, 'UT_BaoBi', 2.0))
    muc_chitra = str(safe_get_profile_value(profile, 'MucChiTra', 'Không rõ'))
    quan_tam = str(safe_get_profile_value(profile, 'QuanTamThienNhien', 'Không'))
    dau_goi_hien_tai = str(safe_get_profile_value(profile, 'LoaiDauGoi', 'Không rõ'))

    is_high_price = any(s in muc_chitra for s in ['200.000', '300.000', '400.000', 'Trên'])
    is_low_price = any(s in muc_chitra for s in ['100.000', '150.000', 'Dưới'])

    # Scenario A: Premium natural
    if (ut_thanhphan > PRIORITY_THRESHOLD_HIGH and
        ut_congdung > PRIORITY_THRESHOLD_HIGH and
        ut_giaca < PRIORITY_THRESHOLD_LOW and
        is_high_price):
        title = "Sản phẩm đề xuất: DẦU GỘI THIÊN NHIÊN CAO CẤP (PREMIUM)"
        details = (
            f"Phân tích: sẵn sàng chi trả cao (Mức chi trả: {muc_chitra}).\n"
            f"Ưu tiên: Thành phần={ut_thanhphan:.2f}, Công dụng={ut_congdung:.2f}, Giá cả={ut_giaca:.2f}.\n"
            "Chiến lược: Dòng thảo dược đặc trị, nhắm phân khúc cao cấp."
        )
        return title, details

    # Scenario B: Affordable natural
    if (ut_giaca > PRIORITY_THRESHOLD_HIGH and is_low_price):
        title = "Sản phẩm đề xuất: DẦU GỘI THIÊN NHIÊN GIÁ RẺ (AFFORDABLE)"
        details = (
            f"Phân tích: Nhạy cảm về giá (Ưu tiên giá={ut_giaca:.2f}), Mức chi trả: {muc_chitra}.\n"
            "Chiến lược: Dòng thiên nhiên cơ bản, nhấn 'An toàn - Tiết kiệm'."
        )
        return title, details

    # Scenario C: Conversion targets
    if (quan_tam.strip().lower() in ['có', 'yes', 'true'] and 'thảo dược' not in dau_goi_hien_tai.lower()):
        title = "Sản phẩm đề xuất: DẦU GỘI THIÊN NHIÊN CẢI TIẾN (CONVERSION)"
        details = (
            f"Phân tích: Quan tâm thiên nhiên nhưng đang dùng sản phẩm công nghiệp.\n"
            f"Ưu tiên: Công dụng={ut_congdung:.2f}, Mùi hương={ut_muihuong:.2f}.\n"
            "Chiến lược: Sản phẩm thiên nhiên với công dụng mạnh và mùi hương hấp dẫn."
        )
        return title, details

    # Scenario D: Fragrance focused
    if (ut_muihuong > PRIORITY_THRESHOLD_HIGH and
        ut_thanhphan < PRIORITY_THRESHOLD_LOW and
        ut_congdung < PRIORITY_THRESHOLD_LOW):
        title = "Sản phẩm đề xuất: DẦU GỘI HƯƠNG NƯỚC HOA (MASS MARKET)"
        details = (
            f"Phân tích: Ưu tiên mùi hương (Mùi={ut_muihuong:.2f}).\n"
            "Chiến lược: Dòng mùi nước hoa, tập trung trải nghiệm cảm quan."
        )
        return title, details

    # Fallback: Balanced
    priorities = {
        'Giá Cả': ut_giaca,
        'Thành Phần': ut_thanhphan,
        'Mùi Hương': ut_muihuong,
        'Công Dụng': ut_congdung,
        'Bao Bì': ut_baobi,
    }
    sorted_priorities = sorted(priorities.items(), key=lambda item: item[1], reverse=True)
    title = "Sản phẩm đề xuất: DẦU GỘI CÂN BẰNG (BALANCED)"
    details = (
        f"Nhóm cân bằng. Các ưu tiên hàng đầu: {sorted_priorities[0][0]} ({sorted_priorities[0][1]:.2f}), "
        f"{sorted_priorities[1][0]} ({sorted_priorities[1][1]:.2f})."
    )
    return title, details


# --- GIAO DIỆN STREAMLIT ---
st.title("💡 Hệ thống Đề xuất Sản phẩm Dầu gội")
st.write("Nhập thông tin của một khách hàng mới để xác định họ thuộc chân dung nào và nhận đề xuất sản phẩm.")

if pipeline is None or cluster_profiles is None:
    st.error("Không thể nạp đầy đủ mô hình/hồ sơ. Hãy chắc chắn rằng `kmeans_pipeline.joblib` và `cluster_profiles.csv` tồn tại trong thư mục ứng dụng.")
    if not MODEL_PATH.exists():
        st.info(f"Tệp mô hình không tìm thấy: {MODEL_PATH}")
    if not PROFILES_PATH.exists():
        st.info(f"Tệp hồ sơ cụm không tìm thấy: {PROFILES_PATH}")
else:
    # Load options from original data if present
    if ORIGINAL_DATA_PATH.exists():
        try:
            df_original = pd.read_csv(ORIGINAL_DATA_PATH)
        except Exception:
            df_original = pd.DataFrame()
    else:
        df_original = pd.DataFrame()

    def opts(col, defaults):
        if col in df_original.columns:
            return df_original[col].dropna().unique().tolist()
        return defaults

    gioitinh_options = opts('GioiTinh', ["Nữ", "Nam", "Khác"])
    nghenghiep_options = opts('NgheNghiep', ["Nhân viên văn phòng", "Sinh viên", "Nội trợ", "Kỹ sư", "Doanh nhân"])
    thunhap_options = opts('ThuNhap', ["Dưới 5 triệu đồng", "5 - 10 triệu đồng", "Trên 10 triệu đồng"])
    mucchi_options = opts('MucChiTra', ["Dưới 100.000 đồng", "100.000 - 150.000 đồng", "150.000 - 200.000 đồng", "200.000 - 300.000 đồng"])
    quantam_options = opts('QuanTamThienNhien', ["Có", "Không"])

    # Sidebar inputs
    st.sidebar.header("Thông tin Khách hàng mới")
    with st.sidebar.form(key='input_form'):
        tuoi = st.slider("Tuổi", 15, 70, 25)
        gioi_tinh = st.selectbox("Giới tính", options=gioitinh_options)
        nghe_nghiep = st.selectbox("Nghề nghiệp", options=nghenghiep_options)
        thu_nhap = st.selectbox("Thu nhập hàng tháng", options=thunhap_options)
        muc_chi_tra = st.selectbox("Mức chi trả cho dầu gội", options=mucchi_options)
        quan_tam_thien_nhien = st.selectbox("Có quan tâm sản phẩm thiên nhiên?", options=quantam_options)

        st.markdown("---")
        st.subheader("Mức độ ưu tiên (1=Không, 3=Rất)")
        ut_giaca = st.slider("Ưu tiên về Giá cả", 1, 3, 2)
        ut_thanhphan = st.slider("Ưu tiên về Thành phần", 1, 3, 2)
        ut_muihuong = st.slider("Ưu tiên về Mùi hương", 1, 3, 2)
        ut_congdung = st.slider("Ưu tiên về Công dụng", 1, 3, 2)
        ut_baobi = st.slider("Ưu tiên về Bao bì", 1, 3, 2)

        submitted = st.form_submit_button("Phân tích và Đề xuất")

    if submitted:
        input_data = pd.DataFrame({
            'Tuoi': [tuoi],
            'GioiTinh': [gioi_tinh],
            'NgheNghiep': [nghe_nghiep],
            'ThuNhap': [thu_nhap],
            'MucChiTra': [muc_chi_tra],
            'QuanTamThienNhien': [quan_tam_thien_nhien],
            'UT_GiaCa': [ut_giaca],
            'UT_ThanhPhan': [ut_thanhphan],
            'UT_MuiHuong': [ut_muihuong],
            'UT_CongDung': [ut_congdung],
            'UT_BaoBi': [ut_baobi]
        })

        try:
            predicted_cluster = pipeline.predict(input_data)[0]
        except Exception as e:
            st.error(f"Lỗi khi dự đoán cụm: {e}")
            predicted_cluster = None

        if predicted_cluster is not None:
            st.success(f"Kết quả phân tích: Khách hàng này thuộc Cụm {predicted_cluster}.")
            rec_title, rec_details = get_product_recommendation(predicted_cluster, cluster_profiles)
            st.subheader(rec_title)
            st.write(rec_details)

            with st.expander(f"Xem hồ sơ chi tiết của Cụm {predicted_cluster}"):
                try:
                    st.dataframe(cluster_profiles.loc[predicted_cluster])
                except Exception:
                    # Fallback for index type mismatch
                    st.dataframe(cluster_profiles)