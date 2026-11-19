import streamlit as st
from PIL import Image
from ultralytics import YOLO


st.set_page_config(page_title="YOLO 이미지 탐지", page_icon="🎯", layout="wide")
st.title("🎯 YOLO 이미지 객체 탐지")
st.caption("이미지를 업로드하면 YOLO 모델이 객체를 탐지하고 바운딩 박스를 표시합니다.")


@st.cache_resource
def load_model(weights_path):
    return YOLO(weights_path)


with st.sidebar:
    st.header("⚙️ 탐지 설정")
    weights_path = st.text_input("모델 가중치 경로", value="yolo26n.pt")
    confidence = st.slider("최소 신뢰도", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    image_width = st.slider("이미지 표시 너비", min_value=300, max_value=1000, value=600, step=50)
    st.caption("공식 가중치 이름 또는 사용자 모델 파일 경로를 입력하세요.")

uploaded_file = st.file_uploader(
    "탐지할 이미지를 업로드하세요",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is None:
    st.info("이미지를 업로드하면 객체 탐지를 시작할 수 있습니다.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader(f"업로드 이미지: {uploaded_file.name}")
    st.image(image, caption="입력 이미지", width=image_width)

    if st.button("🔍 객체 탐지", type="primary"):
        model = load_model(weights_path)
        with st.spinner("모델이 이미지를 분석하고 있습니다..."):
            results = model.predict(source=image, conf=confidence, verbose=False)

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            st.warning("설정한 신뢰도 이상의 객체를 찾지 못했습니다.")
        else:
            annotated_image = result.plot(pil=True)
            st.subheader("탐지 결과")
            st.image(annotated_image, caption="바운딩 박스가 표시된 이미지", width=image_width)

            xywh = boxes.xywh.cpu().tolist()
            xywhn = boxes.xywhn.cpu().tolist()
            xyxy = boxes.xyxy.cpu().tolist()
            xyxyn = boxes.xyxyn.cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()

            detection_rows = []
            for index, class_id in enumerate(class_ids):
                detection_rows.append(
                    {
                        "클래스": result.names[class_id],
                        "신뢰도": round(confidences[index], 4),
                        "중심 x": round(xywh[index][0], 2),
                        "중심 y": round(xywh[index][1], 2),
                        "너비": round(xywh[index][2], 2),
                        "높이": round(xywh[index][3], 2),
                        "xywh 정규화": [round(value, 4) for value in xywhn[index]],
                        "xyxy": [round(value, 2) for value in xyxy[index]],
                        "xyxy 정규화": [round(value, 4) for value in xyxyn[index]],
                    }
                )

            average_confidence = sum(confidences) / len(confidences)
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("탐지 객체 수", len(detection_rows))
            metric_col2.metric("평균 신뢰도", f"{average_confidence:.1%}")

            st.subheader("탐지 정보")
            st.dataframe(detection_rows, hide_index=True, use_container_width=True)