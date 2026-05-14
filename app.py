from __future__ import annotations

import html
import json
import math
import random
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap, MiniMap
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "microhub_hcm_model_ready_dataset.csv"
HOST = "127.0.0.1"
DEFAULT_PORT = 8000

FEATURES_MODEL = [
    "PopulationDensityLevel",
    "RoadNetworkConnectivity",
    "RentCostLevel",
    "OrderDensity",
    "ParkingAvailability",
    "DistanceToCentralHub",
]
FEATURES_FORM = [
    "PopulationDensityLevel",
    "RoadNetworkConnectivity",
    "OrderDensity",
    "ParkingAvailability",
    "DistanceToCentralHub",
]

FEATURE_LABELS = {
    "PopulationDensityLevel": {
        0: "Khu thưa dân",
        1: "Mật độ dân cư trung bình",
        2: "Khu đông dân / nhiều chung cư",
    },
    "RoadNetworkConnectivity": {
        0: "Đường nhỏ, khó tiếp cận",
        1: "Đường rộng vừa, gần đường lớn, dễ tiếp cận",
        2: "Mặt tiền / đường lớn, giao thông thuận lợi",
    },
    "OrderDensity": {
        0: "Ít đơn hàng",
        1: "Mật độ đơn hàng trung bình",
        2: "Rất nhiều đơn hàng",
    },
    "ParkingAvailability": {
        0: "Khó dừng đỗ / khó giao nhận",
        1: "Dừng đỗ ở mức trung bình",
        2: "Dễ dừng đỗ / dễ bốc dỡ hàng",
    },
    "DistanceToCentralHub": {
        0: "Gần kho trung tâm",
        1: "Khoảng cách trung bình",
        2: "Xa kho trung tâm",
    },
}

FEATURE_DISPLAY_NAMES = {
    "PopulationDensityLevel": "Mật độ dân cư",
    "RoadNetworkConnectivity": "Khả năng tiếp cận đường lớn",
    "OrderDensity": "Mật độ đơn hàng",
    "ParkingAvailability": "Khả năng dừng đỗ, bốc dỡ hàng",
    "DistanceToCentralHub": "Khoảng cách đến kho trung tâm",
}


SUITABILITY_LABEL = {
    0: "TỆ",
    1: "TƯƠNG ĐỐI PHÙ HỢP",
    2: "RẤT PHÙ HỢP",
}

SUITABILITY_CLASS = {
    0: "poor",
    1: "acceptable",
    2: "excellent",
}


def split_district_ward(district_ward: str) -> Tuple[str, str]:
    parts = [p.strip() for p in str(district_ward).split(",")]
    if len(parts) == 1:
        return parts[0], parts[0]
    ward = parts[0]
    district = ", ".join(parts[1:])
    return district, ward


def district_sort_key(name: str):
    name = str(name)
    if name.startswith("Quận "):
        try:
            return (0, int(name.replace("Quận ", "").strip()))
        except ValueError:
            return (0, 99)
    order = {
        "Bình Thạnh": 20,
        "Tân Bình": 21,
        "Tân Phú": 22,
        "Gò Vấp": 23,
        "Phú Nhuận": 24,
        "Bình Tân": 25,
        "TP. Thủ Đức": 26,
        "Huyện Bình Chánh": 30,
        "Huyện Nhà Bè": 31,
        "Huyện Hóc Môn": 32,
        "Huyện Củ Chi": 33,
        "Huyện Cần Giờ": 34,
    }
    return (1, order.get(name, 99), name)


def load_data() -> pd.DataFrame:
    df0 = pd.read_csv(DATA_PATH)
    for col in ["Latitude", "Longitude"]:
        df0[col] = pd.to_numeric(df0[col], errors="coerce")
    for col in FEATURES_MODEL + ["RentCost", "Suitability"]:
        df0[col] = pd.to_numeric(df0[col], errors="coerce").astype(int)
    parsed = df0["DistrictWard"].apply(split_district_ward)
    df0["District"] = parsed.apply(lambda x: x[0])
    df0["Ward"] = parsed.apply(lambda x: x[1])
    df0 = df0.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
    df0["RowID"] = df0.index
    return df0


df = load_data()
rent_min = int(df["RentCost"].min())
rent_max = int(df["RentCost"].max())
rent_level_bounds = df.groupby("RentCostLevel")["RentCost"].max().to_dict()
rent_level_0_max = int(rent_level_bounds.get(0, rent_min + (rent_max - rent_min) / 3))
rent_level_1_max = int(rent_level_bounds.get(1, rent_min + 2 * (rent_max - rent_min) / 3))

X = df[FEATURES_MODEL]
y = df["Suitability"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = Perceptron(max_iter=5000, eta0=0.05, random_state=42)
model.fit(X_train, y_train)
model_accuracy = accuracy_score(y_test, model.predict(X_test))
model.fit(X, y)


def normalize_rent_cost(rent_cost: int) -> int:
    if rent_cost <= rent_level_0_max:
        return 0
    if rent_cost <= rent_level_1_max:
        return 1
    return 2


def build_ward_map() -> Dict[str, List[Dict[str, str]]]:
    ward_map0: Dict[str, List[Dict[str, str]]] = {}
    for district in sorted(df["District"].unique(), key=district_sort_key):
        rows = df[df["District"] == district][["Ward", "DistrictWard"]].drop_duplicates()
        rows = rows.sort_values("Ward")
        ward_map0[district] = [
            {"ward": str(r["Ward"]), "districtward": str(r["DistrictWard"])} for _, r in rows.iterrows()
        ]
    return ward_map0


ward_map = build_ward_map()
districts = list(ward_map.keys())


def format_vnd(value: int) -> str:
    return f"{value:,.0f}".replace(",", ".") + " VNĐ/tháng"


def format_vnd_plain(value: int) -> str:
    return f"{value:,.0f}".replace(",", ".")


def parse_rent_cost(value: str) -> int:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if not digits:
        raise ValueError("Vui lòng nhập chi phí thuê mặt bằng bằng số.")
    return int(digits)


def feature_value_text(feature: str, value: int) -> str:
    """Trả về mô tả tiếng Việt, không hiển thị dạng số 0/1/2 trong popup."""
    text = FEATURE_LABELS.get(feature, {}).get(int(value), str(value))
    if " - " in text:
        return text.split(" - ", 1)[1]
    return text


def managerial_assessment(prediction: int, inputs: Dict[str, int], rent_cost: int, rent_level: int) -> str:
    pop = inputs["PopulationDensityLevel"]
    road = inputs["RoadNetworkConnectivity"]
    order = inputs["OrderDensity"]
    parking = inputs["ParkingAvailability"]
    distance = inputs["DistanceToCentralHub"]

    strengths: List[str] = []
    risks: List[str] = []

    if pop == 2:
        strengths.append("mật độ dân cư cao, tạo nền nhu cầu giao hàng ổn định")
    elif pop == 0:
        risks.append("mật độ dân cư thấp, quy mô đơn hàng có thể chưa đủ để khai thác mini-hub")

    if order == 2:
        strengths.append("mật độ đơn hàng cao, phù hợp để gom đơn và rút ngắn chặng cuối")
    elif order == 0:
        risks.append("mật độ đơn hàng thấp, dễ làm tăng chi phí vận hành trên mỗi đơn")

    if road == 2:
        strengths.append("kết nối đường lớn tốt, thuận lợi cho xe tiếp hàng và điều phối shipper")
    elif road == 0:
        risks.append("khả năng tiếp cận đường kém, có thể gây chậm luân chuyển hàng và tăng thời gian giao nhận")

    if parking == 2:
        strengths.append("khả năng dừng đỗ tốt, hỗ trợ bốc dỡ hàng nhanh")
    elif parking == 0:
        risks.append("khả năng dừng đỗ hạn chế, cần kiểm soát giờ nhận hàng hoặc chọn vị trí phụ trợ")

    if distance == 0:
        strengths.append("gần kho trung tâm, thuận lợi cho bổ sung hàng và kiểm soát tồn kho")
    elif distance == 2:
        risks.append("xa kho trung tâm, chi phí tiếp hàng và rủi ro trễ lịch có thể tăng")

    if rent_level == 2:
        risks.append("chi phí thuê ở nhóm cao, cần kiểm tra kỹ doanh thu kỳ vọng và điểm hòa vốn")
    elif rent_level == 0:
        strengths.append("chi phí thuê ở nhóm thấp, giúp giảm áp lực chi phí cố định")

    if prediction == 2:
        return (
            "Vị trí này có mức phù hợp cao về mặt kinh tế - quản trị. "
            f"Điểm mạnh chính gồm: {', '.join(strengths[:4]) if strengths else 'các biến vận hành tương đối cân bằng'}. "
            "Nhà quản trị có thể xem đây là điểm ưu tiên để triển khai mini-hub, nhưng vẫn nên kiểm tra thêm diện tích, pháp lý mặt bằng, giờ cấm tải và chi phí nhân sự trước khi ra quyết định đầu tư."
        )
    if prediction == 1:
        return (
            "Vị trí này ở mức tương đối phù hợp. Có thể khai thác nếu mục tiêu là thử nghiệm thị trường, mở điểm trung chuyển nhỏ hoặc phục vụ một cụm đơn hàng cụ thể. "
            f"Các điểm cần theo dõi gồm: {', '.join(risks[:4]) if risks else 'chi phí thuê và khả năng duy trì mật độ đơn hàng'}. "
            "Khuyến nghị quản trị là triển khai theo mô hình pilot, giới hạn bán kính phục vụ, đo số đơn/ngày và chỉ mở rộng khi chi phí trên mỗi đơn giảm ổn định."
        )
    return (
        "Vị trí này chưa phù hợp để đặt mini-hub ở thời điểm hiện tại. "
        f"Nguyên nhân chính có thể đến từ: {', '.join(risks[:5]) if risks else 'tổ hợp biến đầu vào chưa tạo được lợi thế vận hành rõ ràng'}. "
        "Đề xuất: chưa nên thuê mặt bằng ngay. Nhà quản trị nên cân nhắc chọn phường gần trục đường lớn hơn, tăng khả năng dừng đỗ, tìm vị trí có mật độ đơn hàng cao hơn, hoặc dùng điểm này như điểm giao nhận phụ/locker thay vì mini-hub cố định."
    )


def recommend_location(district: str, district_ward: str, input_vector: Dict[str, int], prediction: int) -> Optional[pd.Series]:
    target = np.array([input_vector[f] for f in FEATURES_MODEL], dtype=float)
    weights = np.array([1.15, 1.10, 0.65, 1.35, 0.70, 1.20])
    pools = []
    same_ward = df[df["DistrictWard"] == district_ward].copy()
    if not same_ward.empty:
        pools.append(same_ward)
    same_district = df[df["District"] == district].copy()
    if not same_district.empty:
        pools.append(same_district)
    pools.append(df.copy())

    for pool in pools:
        pool = pool[pool["Suitability"] >= prediction].copy()
        if pool.empty:
            continue
        arr = pool[FEATURES_MODEL].to_numpy(dtype=float)
        dist = np.sqrt(((arr - target) ** 2 * weights).sum(axis=1))
        pool["_score"] = dist + 0.18 * np.abs(pool["Suitability"] - prediction) - 0.08 * pool["Suitability"]
        return pool.sort_values("_score").iloc[0]
    return None



def recommend_alternative_locations(
    district: str,
    district_ward: str,
    input_vector: Dict[str, int],
    limit: int = 4,
) -> List[Dict[str, object]]:
    """Tìm các địa chỉ thay thế gần giống dữ liệu người dùng nhập nhưng có Suitability tốt hơn."""
    target = np.array([input_vector[f] for f in FEATURES_MODEL], dtype=float)
    weights = np.array([1.15, 1.10, 0.65, 1.35, 0.70, 1.20])
    pools: List[pd.DataFrame] = []

    same_district = df[df["District"] == district].copy()
    if not same_district.empty:
        pools.append(same_district)
    pools.append(df.copy())

    selected_rows: List[Dict[str, object]] = []
    seen: set[int] = set()
    for pool in pools:
        pool = pool[pool["Suitability"] >= 1].copy()
        if pool.empty:
            continue
        arr = pool[FEATURES_MODEL].to_numpy(dtype=float)
        dist = np.sqrt(((arr - target) ** 2 * weights).sum(axis=1))
        pool["_score"] = dist - 0.14 * pool["Suitability"] + np.where(pool["DistrictWard"] == district_ward, -0.10, 0.0)
        for _, rec in pool.sort_values("_score").head(limit * 2).iterrows():
            rowid = int(rec["RowID"])
            if rowid in seen:
                continue
            seen.add(rowid)
            rec_inputs = {
                "PopulationDensityLevel": int(rec["PopulationDensityLevel"]),
                "RoadNetworkConnectivity": int(rec["RoadNetworkConnectivity"]),
                "OrderDensity": int(rec["OrderDensity"]),
                "ParkingAvailability": int(rec["ParkingAvailability"]),
                "DistanceToCentralHub": int(rec["DistanceToCentralHub"]),
            }
            rec_prediction = int(rec["Suitability"])
            rec_rent_level = int(rec["RentCostLevel"])
            selected_rows.append({
                "rowid": rowid,
                "prediction": rec_prediction,
                "css_class": SUITABILITY_CLASS[rec_prediction],
                "full_address": rec["FullAddress"],
                "districtward": rec["DistrictWard"],
                "location_name": rec["LocationName"],
                "lat": round(float(rec["Latitude"]), 6),
                "lon": round(float(rec["Longitude"]), 6),
                "rent": format_vnd(int(rec["RentCost"])),
                "rent_level": rec_rent_level,
                "suitability": SUITABILITY_LABEL[rec_prediction],
                "assessment": managerial_assessment(rec_prediction, rec_inputs, int(rec["RentCost"]), rec_rent_level),
                "road_text": feature_value_text("RoadNetworkConnectivity", int(rec["RoadNetworkConnectivity"])),
                "order_text": feature_value_text("OrderDensity", int(rec["OrderDensity"])),
                "parking_text": feature_value_text("ParkingAvailability", int(rec["ParkingAvailability"])),
            })
            if len(selected_rows) >= limit:
                return selected_rows
    return selected_rows


def add_map_ui_style(m: folium.Map) -> None:
    """Tối ưu vị trí control bản đồ để không bị form nhập dữ liệu che khuất."""
    m.get_root().header.add_child(folium.Element("""
    <style>
        .leaflet-top.leaflet-left {
            left: auto !important;
            right: 18px !important;
            top: 18px !important;
        }
        .leaflet-control-zoom a {
            font-family: "Segoe UI", Arial, sans-serif !important;
            font-weight: 800 !important;
        }
        .leaflet-popup-content {
            font-family: "Segoe UI", Arial, sans-serif !important;
            line-height: 1.42 !important;
        }
        .leaflet-container {
            font-family: "Segoe UI", Arial, sans-serif !important;
        }
        .leaflet-bottom.leaflet-left {
            bottom: 130px !important;
            left: 18px !important;
        }
        .heatspot-dot {
            width: 46px;
            height: 46px;
            border-radius: 999px;
            background: transparent;
            border: 0;
            box-shadow: none;
            cursor: pointer;
            opacity: 0;
        }
    </style>
    """))

def build_demand_hotspots(row: pd.Series, radius_m: int) -> Tuple[List[List[float]], List[Dict[str, object]]]:
    seed_source = f"{row['FullAddress']}-{row['Latitude']:.5f}-{row['Longitude']:.5f}"
    rng = random.Random(seed_source)
    lat0 = float(row["Latitude"])
    lon0 = float(row["Longitude"])
    order = int(row["OrderDensity"])
    pop = int(row["PopulationDensityLevel"])
    cluster_types = [
        "Cụm nhà phố dọc đường",
        "Khu dân cư trong hẻm",
        "Cụm chung cư / căn hộ",
        "Khu văn phòng - thương mại",
        "Điểm gom đơn quanh chợ / cửa hàng",
    ]
    cluster_count = max(4, 4 + order * 3 + pop * 2)
    points: List[List[float]] = []
    centers: List[Dict[str, object]] = []
    lat_meter = 1 / 111_000
    lon_meter = 1 / (111_000 * math.cos(math.radians(lat0)))

    for i in range(cluster_count):
        angle = rng.uniform(0, 2 * math.pi)
        distance = rng.uniform(radius_m * 0.18, radius_m * 0.88)
        center_lat = lat0 + math.sin(angle) * distance * lat_meter
        center_lon = lon0 + math.cos(angle) * distance * lon_meter
        intensity_base = 0.45 + 0.18 * order + 0.12 * pop + rng.uniform(0.0, 0.18)
        order_estimate = int(18 + order * 24 + pop * 12 + rng.uniform(0, 18))
        ctype = rng.choice(cluster_types)
        centers.append(
            {
                "lat": center_lat,
                "lon": center_lon,
                "type": ctype,
                "orders": order_estimate,
                "intensity": round(min(1.0, intensity_base), 2),
            }
        )
        for _ in range(16 + order * 6):
            jitter = abs(rng.gauss(0, radius_m * 0.045))
            jitter_angle = rng.uniform(0, 2 * math.pi)
            p_lat = center_lat + math.sin(jitter_angle) * jitter * lat_meter
            p_lon = center_lon + math.cos(jitter_angle) * jitter * lon_meter
            points.append([p_lat, p_lon, min(1.0, intensity_base + rng.uniform(-0.1, 0.12))])
    return points, centers


def build_map_html(candidate_id: Optional[int] = None, prediction: Optional[int] = None) -> str:
    if candidate_id is None or candidate_id not in set(df["RowID"]):
        m = folium.Map(
            location=[10.7769, 106.7009], zoom_start=11, tiles="CartoDB positron", control_scale=True, prefer_canvas=True
        )
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
        add_map_ui_style(m)
        folium.Marker(
            [10.7769, 106.7009],
            tooltip="TP.HCM",
            popup="Chọn quận, phường và nhập dữ liệu để hệ thống đề xuất vị trí mini-hub.",
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(m)
        MiniMap(toggle_display=True, minimized=True).add_to(m)
        return m.get_root().render()

    row = df.loc[df["RowID"] == candidate_id].iloc[0]
    lat = float(row["Latitude"])
    lon = float(row["Longitude"])
    radius_m = 950 if prediction == 1 else 1350
    m = folium.Map(
        location=[lat, lon], zoom_start=15, tiles="CartoDB positron", control_scale=True, prefer_canvas=True
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    add_map_ui_style(m)
    folium.Circle(
        location=[lat, lon],
        radius=radius_m,
        color="#d39a00",
        weight=2,
        fill=True,
        fill_color="#fff4a8",
        fill_opacity=0.28,
        tooltip=f"Bán kính phục vụ dự kiến: {radius_m:,} m".replace(",", "."),
    ).add_to(m)

    popup_html = f"""
    <div style='font-family:Segoe UI, Arial, sans-serif; width:330px'>
        <h4 style='margin:0 0 8px 0; color:#8a5a00'>📦 Địa chỉ đề xuất đặt Mini-hub</h4>
        <b>{html.escape(str(row['FullAddress']))}</b><br>
        <hr style='margin:8px 0'>
        <b>Khu vực:</b> {html.escape(str(row['DistrictWard']))}<br>
        <b>Tuyến/khu vực:</b> {html.escape(str(row['LocationName']))}<br>
        <b>Giá thuê tham chiếu:</b> {format_vnd(int(row['RentCost']))}<br>
        <b>Mức phù hợp trong dữ liệu:</b> {SUITABILITY_LABEL[int(row['Suitability'])]}<br>
        <b>Mức độ gần đường lớn:</b> {html.escape(feature_value_text('RoadNetworkConnectivity', int(row['RoadNetworkConnectivity'])))}<br>
        <b>Mật độ đơn hàng:</b> {html.escape(feature_value_text('OrderDensity', int(row['OrderDensity'])))}<br>
        <b>Khả năng dừng đỗ, bốc dỡ hàng:</b> {html.escape(feature_value_text('ParkingAvailability', int(row['ParkingAvailability'])))}<br>
    </div>
    """
    folium.Marker(
        [lat, lon],
        tooltip="Mini-hub đề xuất",
        popup=folium.Popup(popup_html, max_width=370, show=True),
        icon=folium.Icon(color="orange", icon="home"),
    ).add_to(m)

    heat_points, centers = build_demand_hotspots(row, radius_m)
    HeatMap(
        heat_points,
        name="Heatmap mật độ đơn hàng",
        radius=34,
        blur=20,
        min_opacity=0.30,
        max_zoom=18,
        gradient={0.15: "#2563eb", 0.35: "#22c55e", 0.58: "#facc15", 0.78: "#f97316", 1.0: "#dc2626"},
    ).add_to(m)

    for idx, c in enumerate(centers, start=1):
        popup = f"""
        <div style='font-family:Segoe UI, Arial, sans-serif; width:285px'>
            <b>Vùng mật độ đơn hàng #{idx}</b><br>
            <b>Đặc điểm khu vực:</b> {c['type']}<br>
            <b>Nhu cầu ước tính:</b> {c['orders']} đơn/ngày<br>
            <b>Mức tập trung đơn hàng:</b> {c['intensity']}<br>
            <small>Gợi ý điều phối: ưu tiên gom đơn theo cụm này và chia tuyến shipper xuất phát từ mini-hub theo bán kính phục vụ.</small>
        </div>
        """
        folium.Marker(
            location=[c["lat"], c["lon"]],
            popup=folium.Popup(popup, max_width=305),
            tooltip=f"Bấm để xem thông tin vùng heatmap #{idx}",
            icon=folium.DivIcon(
                icon_size=(24, 24),
                icon_anchor=(12, 12),
                html='<div class="heatspot-dot"></div>',
            ),
        ).add_to(m)

    MiniMap(toggle_display=True, minimized=True).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m.get_root().render()


def feature_select_html(feature: str, value: int) -> str:
    opts = []
    for v, text in FEATURE_LABELS[feature].items():
        selected = " selected" if int(value) == int(v) else ""
        opts.append(f'<option value="{v}"{selected}>{html.escape(text)}</option>')
    display_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
    return f"""
    <label>
        {html.escape(display_name)}
        <select name="{feature}" required>{''.join(opts)}</select>
    </label>
    """


def alternative_locations_html(alternatives: Optional[List[Dict[str, object]]]) -> str:
    if not alternatives:
        return ""
    cards: List[str] = []
    for idx, item in enumerate(alternatives, start=1):
        payload = html.escape(json.dumps(item, ensure_ascii=False), quote=True)
        cards.append(f"""
            <div class="alternative-card">
                <div class="alt-main">
                    <div class="alt-rank">Gợi ý {idx}</div>
                    <div class="alt-address">{html.escape(str(item['full_address']))}</div>
                    <div class="alt-meta">{html.escape(str(item['location_name']))} · {html.escape(str(item['districtward']))} · {html.escape(str(item['rent']))}</div>
                    <div class="alt-meta">Mức phù hợp trong dữ liệu: {html.escape(str(item['suitability']))}</div>
                </div>
                <button type="button" class="choose-address-btn" onclick="selectAlternativeLocation(JSON.parse(this.dataset.location))" data-location="{payload}">Chọn địa chỉ này</button>
            </div>
        """)
    return f"""
        <section id="alternativeSection" class="alternative-section">
            <div id="selectedRecommendation" class="recommend-section selected-recommendation" style="display:none"></div>
            <div class="alternative-header">
                <div class="section-title">Địa chỉ thay thế đề xuất</div>
                <button type="button" class="mini-toggle-btn" onclick="toggleAlternativeSection()" title="Đóng/Mở địa chỉ thay thế">
                    <span class="close-label">×</span><span class="open-label">Mở</span>
                </button>
            </div>
            <div class="alternative-body">
                <div class="alternative-list">{''.join(cards)}</div>
            </div>
        </section>
    """


def recommendation_html(recommendation: Optional[Dict[str, object]]) -> str:
    if not recommendation:
        return ""
    return f"""
        <section class="recommend-section">
            <div class="section-title">Địa chỉ đề xuất đặt mini-hub</div>
            <div class="address">{html.escape(str(recommendation['full_address']))}</div>
            <div class="muted">{html.escape(str(recommendation['location_name']))} · {html.escape(str(recommendation['districtward']))} · {html.escape(str(recommendation['rent']))}</div>
        </section>
    """


def result_html(result: Optional[Dict[str, object]], error: Optional[str]) -> str:
    if error:
        return f'<section class="result-section error-section"><b>Lỗi dữ liệu nhập:</b> {html.escape(error)}</section>'
    if not result:
        return ""
    return f"""
        <section id="resultSection" class="result-section {html.escape(str(result['css_class']))}">
            <div class="result-left">
                <div>
                    <div class="result-kicker">Kết quả đánh giá</div>
                    <div class="result-title">{html.escape(str(result['label']))}</div>
                </div>
            </div>
            <div class="result-text">{html.escape(str(result['assessment']))}</div>
        </section>
    """


def decision_panel_html(
    result: Optional[Dict[str, object]],
    error: Optional[str],
    recommendation: Optional[Dict[str, object]],
    alternatives: Optional[List[Dict[str, object]]] = None,
) -> str:
    content = alternative_locations_html(alternatives) + recommendation_html(recommendation) + result_html(result, error)
    if not content:
        return ""
    return f"""
    <div id="decisionPanel" class="decision-panel">
        <button type="button" class="toggle-btn decision-toggle" onclick="toggleDecisionPanel()" title="Đóng/Mở kết quả">
            <span class="close-label">×</span><span class="open-label">Mở</span>
        </button>
        <div class="decision-content">{content}</div>
    </div>
    """


def render_index(
    selected_district: Optional[str] = None,
    selected_districtward: Optional[str] = None,
    form_values: Optional[Dict[str, int]] = None,
    result: Optional[Dict[str, object]] = None,
    recommendation: Optional[Dict[str, object]] = None,
    alternatives: Optional[List[Dict[str, object]]] = None,
    error: Optional[str] = None,
    collapsed: bool = False,
    map_url: str = "/map",
) -> str:
    selected_district = selected_district or (districts[0] if districts else "")
    selected_districtward = selected_districtward or (ward_map[selected_district][0]["districtward"] if selected_district else "")
    form_values = form_values or {
        "PopulationDensityLevel": 1,
        "RoadNetworkConnectivity": 1,
        "OrderDensity": 1,
        "ParkingAvailability": 1,
        "DistanceToCentralHub": 1,
        "RentCost": int((rent_min + rent_max) / 2),
    }
    selects = "\n".join(feature_select_html(f, int(form_values.get(f, 1))) for f in FEATURES_FORM)
    collapsed_class = "collapsed" if collapsed else ""
    html_doc = TEMPLATE
    replacements = {
        "@@MAP_URL@@": html.escape(map_url, quote=True),
        "@@COLLAPSED_CLASS@@": collapsed_class,
        "@@RENT_MIN@@": str(rent_min),
        "@@RENT_MAX@@": str(rent_max),
        "@@RENT_VALUE@@": format_vnd_plain(int(form_values.get("RentCost", int((rent_min + rent_max) / 2)))),
        "@@RENT_MIN_TEXT@@": format_vnd(rent_min),
        "@@RENT_MAX_TEXT@@": format_vnd(rent_max),
        "@@RENT_LEVEL_0_MAX_TEXT@@": format_vnd(rent_level_0_max),
        "@@RENT_LEVEL_1_MAX_TEXT@@": format_vnd(rent_level_1_max),
        "@@FEATURE_SELECTS@@": selects,
        "@@WARD_MAP_JSON@@": json.dumps(ward_map, ensure_ascii=False),
        "@@SELECTED_DISTRICT_JSON@@": json.dumps(selected_district, ensure_ascii=False),
        "@@SELECTED_DISTRICTWARD_JSON@@": json.dumps(selected_districtward, ensure_ascii=False),
        "@@MODEL_ACCURACY@@": f"{model_accuracy:.2%}",
        "@@DECISION_PANEL_HTML@@": decision_panel_html(result, error, recommendation, alternatives),
    }
    for key, val in replacements.items():
        html_doc = html_doc.replace(key, val)
    return html_doc


def evaluate_form(params: Dict[str, List[str]]) -> str:
    selected_district = params.get("district", [districts[0]])[0]
    selected_districtward = params.get("districtward", [ward_map[selected_district][0]["districtward"]])[0]
    form_values = {f: int(params.get(f, [1])[0]) for f in FEATURES_FORM}
    error = None
    result = None
    recommendation = None
    alternatives = None
    map_url = "/map"
    collapsed = False
    try:
        rent_cost = parse_rent_cost(params.get("RentCost", [""])[0])
        form_values["RentCost"] = rent_cost
        if rent_cost < rent_min or rent_cost > rent_max:
            error = f"Giá thuê phải nằm trong khoảng {format_vnd(rent_min)} đến {format_vnd(rent_max)} theo phạm vi dữ liệu gốc."
        else:
            rent_level = normalize_rent_cost(rent_cost)
            input_vector = {
                "PopulationDensityLevel": form_values["PopulationDensityLevel"],
                "RoadNetworkConnectivity": form_values["RoadNetworkConnectivity"],
                "RentCostLevel": rent_level,
                "OrderDensity": form_values["OrderDensity"],
                "ParkingAvailability": form_values["ParkingAvailability"],
                "DistanceToCentralHub": form_values["DistanceToCentralHub"],
            }
            prediction = int(model.predict(pd.DataFrame([input_vector], columns=FEATURES_MODEL))[0])
            district, _ward = split_district_ward(selected_districtward)
            assessment = managerial_assessment(prediction, form_values, rent_cost, rent_level)
            result = {
                "prediction": prediction,
                "label": SUITABILITY_LABEL[prediction],
                "css_class": SUITABILITY_CLASS[prediction],
                "assessment": assessment,
                "rent_level": rent_level,
                "rent_text": format_vnd(rent_cost),
            }
            if prediction in (1, 2):
                rec = recommend_location(district, selected_districtward, input_vector, prediction)
                if rec is not None:
                    recommendation = {
                        "rowid": int(rec["RowID"]),
                        "full_address": rec["FullAddress"],
                        "districtward": rec["DistrictWard"],
                        "location_name": rec["LocationName"],
                        "lat": round(float(rec["Latitude"]), 6),
                        "lon": round(float(rec["Longitude"]), 6),
                        "rent": format_vnd(int(rec["RentCost"])),
                        "suitability": SUITABILITY_LABEL[int(rec["Suitability"])],
                    }
                    map_url = f"/map?candidate_id={recommendation['rowid']}&prediction={prediction}"
            else:
                alternatives = recommend_alternative_locations(district, selected_districtward, input_vector, limit=4)
            collapsed = True
    except Exception as exc:  # noqa: BLE001
        error = f"Dữ liệu nhập chưa hợp lệ: {exc}"
    return render_index(
        selected_district=selected_district,
        selected_districtward=selected_districtward,
        form_values=form_values,
        result=result,
        recommendation=recommendation,
        alternatives=alternatives,
        error=error,
        collapsed=collapsed,
        map_url=map_url,
    )


TEMPLATE = r'''<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Mini-hub Recommendation | TP.HCM</title>
    <style>
        :root {
            --bg: #f8fafc;
            --ink: #061224;
            --muted: #64748b;
            --card: #ffffff;
            --border: rgba(15, 23, 42, 0.12);
            --accent: #0ea5e9;
            --blue: #0b7cff;
            --danger: #ef4444;
            --success: #16a34a;
            --shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
        }
        * { box-sizing: border-box; }
        html, body {
            height: 100%;
            margin: 0;
            font-family: "Segoe UI", Arial, Helvetica, sans-serif;
            color: var(--ink);
            background: var(--bg);
            -webkit-font-smoothing: antialiased;
            text-rendering: optimizeLegibility;
        }
        .map-frame { position: fixed; inset: 0; width: 100vw; height: 100vh; border: 0; z-index: 1; background: #eaf3f2; }
        .control-panel {
            position: fixed;
            z-index: 10;
            top: 18px;
            left: 18px;
            width: min(520px, calc(100vw - 36px));
            max-height: calc(100vh - 36px);
            overflow-y: auto;
            background: #ffffff;
            backdrop-filter: none;
            border: 1px solid var(--border);
            border-radius: 24px;
            box-shadow: var(--shadow);
            padding: 18px;
            transition: width .25s ease, max-height .25s ease, padding .25s ease, border-radius .25s ease;
        }
        .control-panel.collapsed { width: 118px; max-height: 64px; overflow: hidden; padding: 14px; border-radius: 20px; }
        .control-panel.collapsed .input-form, .control-panel.collapsed .brand-logo-wrap { display: none; }
        .panel-header { position: relative; display: flex; justify-content: center; gap: 14px; align-items: flex-start; margin-bottom: 14px; }
        .brand-logo-wrap {
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .brand-logo {
            width: 238px;
            max-width: 100%;
            height: auto;
            display: block;
            border-radius: 18px;
            filter: none;
        }
        .input-toggle { position: absolute; top: 0; right: 0; }
        p { margin: 0; color: var(--muted); font-size: 13.5px; line-height: 1.45; }
        .toggle-btn {
            cursor: pointer;
            border: 0;
            min-width: 54px;
            height: 42px;
            border-radius: 999px;
            font-weight: 900;
            font-size: 16px;
            color: white;
            background: var(--danger);
            box-shadow: 0 10px 22px rgba(239, 68, 68, .20);
        }
        .toggle-btn .open-label { display: none; }
        .control-panel.collapsed .toggle-btn, .decision-panel.collapsed .toggle-btn {
            background: var(--success);
            box-shadow: 0 10px 22px rgba(22, 163, 74, .20);
        }
        .control-panel.collapsed .toggle-btn .close-label, .decision-panel.collapsed .toggle-btn .close-label { display: none; }
        .control-panel.collapsed .toggle-btn .open-label, .decision-panel.collapsed .toggle-btn .open-label { display: inline; }
        .input-form { display: grid; gap: 13px; }
        .grid-two { display: grid; grid-template-columns: 1fr 1fr; gap: 11px; }
        label { display: grid; gap: 7px; font-size: 13px; font-weight: 800; color: #334155; }
        select, input {
            width: 100%;
            border: 1.5px solid #8bc5ff;
            border-radius: 14px;
            padding: 12px 13px;
            font-size: 15px;
            font-family: "Segoe UI", Arial, Helvetica, sans-serif;
            background: linear-gradient(135deg, #f7fdff 0%, #e6f8ff 100%);
            color: #172033;
            outline: none;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.85), 0 4px 12px rgba(14, 165, 233, .08);
        }
        select:focus, input:focus {
            border-color: #1684ff;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, .14), inset 0 1px 0 rgba(255,255,255,.9);
        }
        .field-help { color: #64748b; font-size: 12.5px; font-weight: 600; line-height: 1.4; }
        .model-note { padding: 10px 12px; border-radius: 14px; background: #eff6ff; border: 1px solid #bfdbfe; color: #1e3a8a; font-size: 12.5px; line-height: 1.45; text-align: center; }
        .primary-btn {
            cursor: pointer;
            border: 0;
            border-radius: 16px;
            font-weight: 850;
            letter-spacing: .01em;
            padding: 14px 16px;
            background: linear-gradient(135deg, #005bea, #00c6fb);
            color: white;
            font-size: 15px;
            box-shadow: 0 12px 26px rgba(0, 91, 234, .24);
        }
        .decision-panel {
            position: fixed;
            z-index: 8;
            left: 18px;
            right: 18px;
            bottom: 18px;
            background: rgba(255,255,255,.965);
            backdrop-filter: blur(14px);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            border-radius: 24px;
            padding: 16px 76px 16px 18px;
            transition: width .25s ease, height .25s ease, padding .25s ease, border-radius .25s ease;
        }
        .decision-toggle { position: absolute; top: 14px; right: 14px; }
        .decision-panel.collapsed {
            left: auto;
            right: 18px;
            bottom: 18px;
            width: 118px;
            height: 64px;
            padding: 12px;
            border-radius: 20px;
            overflow: hidden;
        }
        .decision-panel.collapsed .decision-content { display: none; }
        .decision-panel.collapsed .decision-toggle { position: static; width: 100%; }
        .decision-content { display: grid; gap: 12px; }
        .alternative-section {
            border-radius: 18px;
            padding: 14px 16px;
            background: #f8fafc;
            border: 1px solid var(--border);
        }
        .alternative-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
        .alternative-body { margin-top: 12px; }
        .alternative-section.collapsed .alternative-body { display: none; }
        .mini-toggle-btn {
            cursor: pointer;
            border: 0;
            min-width: 46px;
            height: 34px;
            border-radius: 999px;
            font-weight: 950;
            font-size: 14px;
            color: #ffffff;
            background: var(--danger);
            box-shadow: 0 8px 18px rgba(239, 68, 68, .16);
        }
        .mini-toggle-btn .open-label { display: none; }
        .alternative-section.collapsed .mini-toggle-btn { background: var(--success); box-shadow: 0 8px 18px rgba(22, 163, 74, .16); }
        .alternative-section.collapsed .mini-toggle-btn .close-label { display: none; }
        .alternative-section.collapsed .mini-toggle-btn .open-label { display: inline; }
        .alternative-list { display: grid; gap: 10px; }
        .alternative-card {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 12px;
            align-items: center;
            padding: 12px;
            border-radius: 16px;
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, .10);
        }
        .alt-rank { color: #b45309; font-size: 12px; font-weight: 900; text-transform: uppercase; letter-spacing: .03em; }
        .alt-address { font-size: 16px; line-height: 1.3; font-weight: 900; margin-top: 3px; }
        .alt-meta { color: var(--muted); font-size: 12.5px; margin-top: 2px; }
        .choose-address-btn {
            cursor: pointer;
            border: 0;
            border-radius: 999px;
            padding: 11px 14px;
            font-family: "Segoe UI", Arial, Helvetica, sans-serif;
            font-size: 13px;
            font-weight: 900;
            color: #ffffff;
            background: var(--success);
            box-shadow: 0 10px 18px rgba(22, 163, 74, .18);
            white-space: nowrap;
        }
        .selected-recommendation { margin-bottom: 12px; }
        .recommend-section {
            border-radius: 18px;
            padding: 14px 16px;
            background: #f0f8ff;
            border: 1px solid #bfdbfe;
        }
        .section-title { font-size: 13px; color: #075da8; font-weight: 900; }
        .address { font-size: 20px; font-weight: 900; margin-top: 5px; letter-spacing: -0.01em; }
        .muted { color: var(--muted); font-size: 13px; margin-top: 3px; }
        .result-section {
            display: grid;
            grid-template-columns: minmax(260px, 360px) 1fr;
            gap: 20px;
            align-items: center;
            border-radius: 22px;
            padding: 18px 20px;
            background: linear-gradient(135deg, rgba(255,255,255,.98), rgba(248,250,252,.96));
            border: 1px solid rgba(15, 23, 42, .10);
            box-shadow: 0 14px 34px rgba(15, 23, 42, .08);
            position: relative;
            overflow: hidden;
        }
        .result-section::before { display: none; }
        .result-left { display: flex; gap: 14px; align-items: center; min-width: 0; }
        .result-icon { display: none; }
        .result-kicker { font-size: 12px; font-weight: 950; color: #64748b; text-transform: uppercase; letter-spacing: .06em; }
        .result-title { font-size: 30px; line-height: 1.08; font-weight: 950; margin-top: 4px; letter-spacing: -0.02em; }
        .result-text { line-height: 1.72; color: #1e293b; font-size: 15.5px; }
        .result-section.poor {
            background: linear-gradient(135deg, #fff7f7 0%, #ffe9ec 100%);
            border-color: rgba(220, 38, 38, .30);
            box-shadow: 0 14px 34px rgba(220, 38, 38, .10);
        }
        .result-section.poor .result-kicker,
        .result-section.poor .result-title,
        .result-section.poor .result-text { color: #8f1d1d; }
        .result-section.acceptable {
            background: linear-gradient(135deg, #fffaf0 0%, #fff0d5 100%);
            border-color: rgba(245, 158, 11, .34);
            box-shadow: 0 14px 34px rgba(245, 158, 11, .12);
        }
        .result-section.acceptable .result-kicker,
        .result-section.acceptable .result-title,
        .result-section.acceptable .result-text { color: #9a3f10; }
        .result-section.excellent {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border-color: rgba(22, 101, 52, .34);
            box-shadow: 0 14px 34px rgba(22, 101, 52, .12);
        }
        .result-section.excellent .result-kicker,
        .result-section.excellent .result-title,
        .result-section.excellent .result-text { color: #14532d; }
        .error-section { border-left: 8px solid #ef4444; color: #7f1d1d; }
        @media (max-width: 860px) {
            .grid-two { grid-template-columns: 1fr; }
            .result-section { grid-template-columns: 1fr; }
            .decision-panel { max-height: 42vh; overflow-y: auto; padding-right: 70px; }
            .control-panel { width: min(520px, calc(100vw - 28px)); left: 14px; top: 14px; }
            .alternative-card { grid-template-columns: 1fr; }
            .choose-address-btn { width: 100%; }
        }
    </style>
</head>
<body>
    <iframe id="mapFrame" class="map-frame" src="@@MAP_URL@@"></iframe>
    <div id="inputPanel" class="control-panel @@COLLAPSED_CLASS@@">
        <div class="panel-header">
            <div class="brand-logo-wrap">
                <img class="brand-logo" src="/assets/nexushub_logo.png" alt="NexusHub logo">
            </div>
            <button type="button" class="toggle-btn input-toggle" onclick="togglePanel()" title="Đóng/Mở form"><span class="close-label">×</span><span class="open-label">Mở</span></button>
        </div>
        <form method="POST" action="/evaluate" class="input-form">
            <div class="grid-two">
                <label>Quận / Thành phố<select name="district" id="districtSelect"></select></label>
                <label>Phường / Xã<select name="districtward" id="wardSelect"></select></label>
            </div>
            <label>Chi phí thuê mặt bằng mong muốn, VNĐ/tháng
                <input id="rentCostInput" name="RentCost" type="text" inputmode="numeric" data-min="@@RENT_MIN@@" data-max="@@RENT_MAX@@" value="@@RENT_VALUE@@" placeholder="Ví dụ: 30.000.000" required>
            </label>
            @@FEATURE_SELECTS@@
            <button type="submit" class="primary-btn">Đánh giá</button>
        </form>
    </div>
    @@DECISION_PANEL_HTML@@
    <script>
        const wardMap = @@WARD_MAP_JSON@@;
        const selectedDistrict = @@SELECTED_DISTRICT_JSON@@;
        const selectedDistrictWard = @@SELECTED_DISTRICTWARD_JSON@@;
        const districtSelect = document.getElementById('districtSelect');
        const wardSelect = document.getElementById('wardSelect');
        function populateDistricts() {
            districtSelect.innerHTML = '';
            Object.keys(wardMap).forEach(d => {
                const opt = document.createElement('option');
                opt.value = d;
                opt.textContent = d;
                if (d === selectedDistrict) opt.selected = true;
                districtSelect.appendChild(opt);
            });
            populateWards();
        }
        function populateWards() {
            const district = districtSelect.value;
            wardSelect.innerHTML = '';
            (wardMap[district] || []).forEach(item => {
                const opt = document.createElement('option');
                opt.value = item.districtward;
                opt.textContent = item.ward;
                if (item.districtward === selectedDistrictWard) opt.selected = true;
                wardSelect.appendChild(opt);
            });
        }
        function togglePanel() { document.getElementById('inputPanel').classList.toggle('collapsed'); }
        function toggleDecisionPanel() {
            const panel = document.getElementById('decisionPanel');
            if (panel) panel.classList.toggle('collapsed');
        }
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text ?? '';
            return div.innerHTML;
        }
        function toggleAlternativeSection() {
            const section = document.getElementById('alternativeSection');
            if (section) section.classList.toggle('collapsed');
        }
        function updateResultForAlternative(info) {
            const result = document.getElementById('resultSection');
            if (!result) return;
            result.classList.remove('poor', 'acceptable', 'excellent');
            result.classList.add(info.css_class || (Number(info.prediction) === 2 ? 'excellent' : 'acceptable'));
            result.innerHTML = `
                <div class="result-left">
                    <div>
                        <div class="result-kicker">Kết quả đánh giá sau khi chọn địa chỉ mới</div>
                        <div class="result-title">${escapeHtml(info.suitability)}</div>
                    </div>
                </div>
                <div class="result-text">${escapeHtml(info.assessment || 'Địa chỉ mới có mức phù hợp tốt hơn dữ liệu vừa nhập và có thể được dùng làm phương án đề xuất để nhà quản trị cân nhắc triển khai mini-hub.')}</div>
            `;
        }
        function selectAlternativeLocation(info) {
            const mapFrame = document.getElementById('mapFrame');
            mapFrame.src = `/map?candidate_id=${encodeURIComponent(info.rowid)}&prediction=${encodeURIComponent(info.prediction)}`;
            const section = document.getElementById('alternativeSection');
            const box = document.getElementById('selectedRecommendation');
            if (box) {
                box.style.display = 'block';
                box.innerHTML = `
                    <div class="section-title">Địa chỉ đề xuất đặt mini-hub</div>
                    <div class="address">${escapeHtml(info.full_address)}</div>
                    <div class="muted">${escapeHtml(info.location_name)} · ${escapeHtml(info.districtward)} · ${escapeHtml(info.rent)}</div>
                    <div class="muted">Đánh giá mới: ${escapeHtml(info.suitability)}</div>
                `;
            }
            if (section) section.classList.add('collapsed');
            updateResultForAlternative(info);
            const panel = document.getElementById('decisionPanel');
            if (panel) panel.classList.remove('collapsed');
        }
        function formatMoneyInput(el) {
            const raw = (el.value || '').replace(/\D/g, '');
            el.value = raw.replace(/\B(?=(\d{3})+(?!\d))/g, '.');
        }
        const rentCostInput = document.getElementById('rentCostInput');
        if (rentCostInput) {
            formatMoneyInput(rentCostInput);
            rentCostInput.addEventListener('input', () => formatMoneyInput(rentCostInput));
        }
        districtSelect.addEventListener('change', populateWards);
        populateDistricts();
    </script>
</body>
</html>'''


class MicroHubHandler(BaseHTTPRequestHandler):
    def send_html(self, content: str, status: int = 200):
        encoded = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "":
            self.send_html(render_index())
        elif parsed.path == "/map":
            query = urllib.parse.parse_qs(parsed.query)
            candidate_id = query.get("candidate_id", [None])[0]
            prediction = query.get("prediction", [None])[0]
            candidate_id_int = int(candidate_id) if candidate_id is not None else None
            prediction_int = int(prediction) if prediction is not None else None
            self.send_html(build_map_html(candidate_id_int, prediction_int))
        elif parsed.path.startswith("/assets/"):
            asset_path = (BASE_DIR / parsed.path.lstrip("/")).resolve()
            assets_dir = (BASE_DIR / "assets").resolve()
            if assets_dir in asset_path.parents and asset_path.exists():
                content = asset_path.read_bytes()
                self.send_response(200)
                if asset_path.suffix.lower() == ".png":
                    self.send_header("Content-Type", "image/png")
                else:
                    self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_html("<h1>404</h1><p>Không tìm thấy tài nguyên.</p>", status=404)
        else:
            self.send_html("<h1>404</h1><p>Không tìm thấy trang.</p>", status=404)

    def do_POST(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/evaluate":
            self.send_html("<h1>404</h1><p>Không tìm thấy trang.</p>", status=404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        params = urllib.parse.parse_qs(body)
        self.send_html(evaluate_form(params))

    def log_message(self, format: str, *args):  # noqa: A002
        print("[%s] %s" % (self.log_date_time_string(), format % args))


def find_free_port(start: int = DEFAULT_PORT, attempts: int = 20) -> int:
    import socket
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((HOST, port)) != 0:
                return port
    return start


def main():
    port = find_free_port()
    server = ThreadingHTTPServer((HOST, port), MicroHubHandler)
    url = f"http://{HOST}:{port}"
    print("=" * 70)
    print("AI-powered Last Mile Delivery Micro-hub Recommendation System")
    print(f"Dataset: {DATA_PATH}")
    print(f"RentCost range: {format_vnd(rent_min)} - {format_vnd(rent_max)}")
    print(f"Perceptron test accuracy: {model_accuracy:.2%}")
    print(f"Open app: {url}")
    print("Press CTRL + C to stop.")
    print("=" * 70)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDa dung server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
