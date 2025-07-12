import streamlit as st
import cv2
import numpy as np
from fpdf import FPDF
import time
import os
import io
import sys
from collections import defaultdict
from ultralytics import YOLO
import tempfile
import pandas as pd
from ui_helpers import inject_custom_css, styled_header, display_total, display_bill

# Set page configuration
st.set_page_config(
    page_title="Automated Grocery Billing",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'persistent_items' not in st.session_state:
    st.session_state.persistent_items = []
if 'tracked_instances' not in st.session_state:
    st.session_state.tracked_instances = []
if 'frame_rate_buffer' not in st.session_state:
    st.session_state.frame_rate_buffer = []
if 'avg_frame_rate' not in st.session_state:
    st.session_state.avg_frame_rate = 0
if 'last_detection_time' not in st.session_state:
    st.session_state.last_detection_time = {}
if 'detection_cooldown' not in st.session_state:
    st.session_state.detection_cooldown = 3  # seconds
if 'stop_detection' not in st.session_state:
    st.session_state.stop_detection = False
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = "default_uploader"

# Product prices and classes
classes = {
    '5Star':10,
    'Coco-Cola':10,
    'Dairy milk':10,
    'Dettol':56,
    'Green Tea':121,
    'KitKat':15,
    'KurKure':25,
    'Lays':15,
    'Park Avenue':93,
    'Peanut Butter':55,
    'Pril':30,
    'Sandal Soap':40,
    'ThumsUp':25,
}

# Colors for bounding boxes
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106)]

# Constants for tracking
max_distance = 50  # pixels between centers to consider same object
instance_timeout = 3  # seconds before removing unseen instances
fps_avg_len = 200

def load_model(model_path):
    """Load the YOLO model"""
    try:
        model = YOLO(model_path, task='detect')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def add_to_bill(class_name):
    """Add an item to the bill with cooldown to avoid multiple counts"""
    current_time = time.time()
    last_time = st.session_state.last_detection_time.get(class_name, 0)
    
    if current_time - last_time > st.session_state.detection_cooldown:
        st.session_state.persistent_items.append(class_name)
        st.session_state.last_detection_time[class_name] = current_time
        return True
    return False

def process_frame(frame, model, min_thresh=0.5):
    """Process a single frame for object detection"""
    current_time = time.time()
    
    # Run inference
    results = model(frame, verbose=False)
    detections = results[0].boxes
    
    # Cleanup old tracked instances
    st.session_state.tracked_instances = [
        instance for instance in st.session_state.tracked_instances 
        if current_time - instance['last_seen'] <= instance_timeout
    ]
    
    # Process detections
    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        classidx = int(detections[i].cls.item())
        classname = model.names[classidx]
        conf = detections[i].conf.item()
        
        if conf > min_thresh:
            # Draw bounding box
            color = bbox_colors[classidx % len(bbox_colors)]
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            
            # Add label
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(xyxy[1], labelSize[1] + 10)
            cv2.rectangle(frame, (xyxy[0], label_ymin-labelSize[1]-10), 
                         (xyxy[0]+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xyxy[0], label_ymin-7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Calculate center of detection
            x_center = (xyxy[0] + xyxy[2]) // 2
            y_center = (xyxy[1] + xyxy[3]) // 2
            
            # Check against existing tracked instances
            matched = False
            for instance in st.session_state.tracked_instances:
                if instance['class'] == classname:
                    # Calculate distance between centers
                    dx = x_center - instance['position'][0]
                    dy = y_center - instance['position'][1]
                    distance = (dx**2 + dy**2)**0.5
                    
                    if distance < max_distance:
                        # Update existing instance
                        instance['position'] = (x_center, y_center)
                        instance['last_seen'] = current_time
                        matched = True
                        break
            
            if not matched and conf > 0.70:  # Higher confidence for new items
                # Add new instance
                st.session_state.tracked_instances.append({
                    'class': classname,
                    'position': (x_center, y_center),
                    'last_seen': current_time,
                    'id': len(st.session_state.tracked_instances) + 1
                })
                # Add to bill
                add_to_bill(classname)
    
    # Display FPS
    if len(st.session_state.frame_rate_buffer) > 0:
        cv2.putText(frame, f'FPS: {st.session_state.avg_frame_rate:0.2f}', (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return frame

def generate_pdf_bill(bill_data, total_amount):
    # Define details
    shop_name = "Grocer Store"
    shop_address = "123 Tech Street, Somewhere\nPhone: +91 98765 43210\nEmail: info@groceryshop.com"
    date_str = time.strftime("%Y-%m-%d %H:%M:%S")
    bill_id = f"Bill-{int(time.time())}"
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Color scheme
    header_color = (79, 129, 189)  # Blue
    accent_color = (220, 53, 69)   # Red
    text_color = (51, 51, 51)      # Dark gray
    line_color = (200, 200, 200)   # Light gray
    
    # Header Section
    pdf.set_font("Arial", "B", 24)
    pdf.set_text_color(*header_color)
    pdf.cell(0, 15, shop_name, ln=True, align="C")
    
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(*text_color)
    pdf.multi_cell(0, 5, shop_address, align="C")
    pdf.ln(10)
    
    # Bill Info
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Bill ID: {bill_id}", ln=True)
    pdf.cell(0, 8, f"Date: {date_str}", ln=True)
    pdf.ln(15)
    
    # Column headers
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(*header_color)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, "Item", ln=False)
    pdf.set_x(120)
    pdf.cell(20, 10, "Qty", ln=False)
    pdf.set_x(140)
    pdf.cell(25, 10, "Unit Price", ln=False)
    pdf.set_x(170)
    pdf.cell(20, 10, "Total", ln=True)
    
    # Draw header underline
    pdf.set_draw_color(*header_color)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)
    
    # Bill Items
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(*text_color)
    line_height = 10
    fill = False
    
    for idx, row in enumerate(bill_data):
        # Alternate row background
        if fill:
            pdf.set_fill_color(245, 245, 245)
            pdf.rect(10, pdf.get_y(), 190, line_height, 'F')
        
        # Item Name
        pdf.cell(0, line_height, row["Item"], ln=False)
        
        # Quantity
        pdf.set_x(120)
        pdf.cell(20, line_height, str(row["Quantity"]), ln=False)
        
        # Unit Price
        pdf.set_x(140)
        pdf.cell(25, line_height, row["Unit Price"], ln=False)
        
        # Total
        pdf.set_x(170)
        pdf.cell(20, line_height, row["Total"], ln=True)
        
        # Draw line separator
        pdf.set_draw_color(*line_color)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)
        
        fill = not fill
    
    # Total Section
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(*accent_color)
    pdf.cell(0, 10, "Total Amount:", ln=False)
    pdf.set_x(170)
    pdf.cell(20, 10, f"Rs.{total_amount}", ln=True)
    
    # Footer
    pdf.ln(15)
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 5, 
        "Thank you for choosing Smart Grocery!\n"
        "Terms: Goods sold are non-refundable | Warranty: 7 days", 
        align="C")
    
    pdf_data = pdf.output(dest="S").encode("latin1")
    return pdf_data

def toggle_stop_detection():
    """Toggle the stop detection flag"""
    st.session_state.stop_detection = not st.session_state.stop_detection

def main():
    inject_custom_css()
    
    # Title with logo
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 30px;">
        <h1 style="margin: 0;">🛒 Smart Grocery Checkout</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Settings")
        
        # Model selection
        model_path = st.text_input("Model Path", "my_model.pt", key="model_path_input")
        
        # Video source
        source_options = ["Upload Video", "Webcam"]
        source_type = st.radio("Select Source", source_options, key="source_type_radio")
        
        # Confidence threshold
        min_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05, key="conf_thresh_slider")
        
        # Resolution
        resolution_options = ["640x480", "720x680", "1280x720"]
        resolution = st.selectbox("Resolution", resolution_options, key="resolution_select")
        resW, resH = map(int, resolution.split("x"))
        
        # Detection cooldown
        st.session_state.detection_cooldown = st.slider(
            "Detection Cooldown (seconds)", 
            1, 10, 3, 
            help="Time to wait before counting the same object again",
            key="cooldown_slider"
        )
        
        # Load model button
        if st.button("Load Model and Start", key="load_model_button"):
            st.session_state.stop_detection = False
            if os.path.exists(model_path):
                st.session_state.model = load_model(model_path)
                if st.session_state.model:
                    st.success("Model loaded successfully!")
            else:
                st.error("Model path is invalid or model was not found.")
        
        #if st.button("Stop Detection", key="stop_detection_button", on_click=toggle_stop_detection):
        #    pass
    
    # Main content area with two columns
    col1, col2 = st.columns([0.6, 0.4])
    frame_placeholder = None
    
    with col1:
        st.header("Automated Billing System")
        if 'model' in st.session_state and st.session_state.model and 'cap' in st.session_state:
            frame_placeholder = st.empty()
        else:
            st.markdown("""
            <div class="empty-video-placeholder">
                <div class="icon">📷</div>
                <h3 style="margin: 0; color: var(--primaryColor);">Video Feed Preview</h3>
                <p style="margin: 0.5rem 0 0; opacity: 0.8;">Load model and start detection to begin</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Stop button under video feed
        if st.button("Stop Detection", key="stop_detection_button", on_click=toggle_stop_detection):
            pass
        
        # Upload video file or use webcam
        if source_type == "Upload Video":
            uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"], key=st.session_state.get("uploader_key","default_uploader"))
            if uploaded_file is not None:
                if 'cap' not in st.session_state or st.session_state.current_source != "upload":
                    # Save uploaded file to temp location
                    tfile = tempfile.NamedTemporaryFile(delete=False) 
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
                    if 'cap' in st.session_state:
                        st.session_state.cap.release()
                    st.session_state.cap = cv2.VideoCapture(video_path)
                    st.session_state.current_source="upload"
        else:  # Webcam
            if 'cap' not in st.session_state or st.session_state.current_source != "webcam":
                if 'cap' in st.session_state:
                    st.session_state.cap.release()
                st.session.cap = cv2.VideoCapture(0)
                st.session.cap.set(3, resW)
                st.session.cap.set(4, resH)
                st.session.current_source="webcam"
    
    with col2:
        styled_header("Shopping Cart")
        
        
        #bill_placeholder = st.empty()
        if 'model' in st.session_state and st.session_state.model and 'cap' in st.session_state:
            bill_placeholder = st.empty()
        else:
            st.markdown("""
            <div class="empty-bill-placeholder">
                <div class="icon">🛒</div>
                <h4 style="margin: 0;">Empty Shopping Cart</h4>
                <p style="margin: 0.5rem 0 0; opacity: 0.8;">Detected items will appear here</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Total amount display
        #total_placeholder = st.empty()
        if 'model' in st.session_state and st.session_state.model and 'cap' in st.session_state:
            total_placeholder = st.empty()
        else:
            st.markdown("""
            <div class="empty-bill-placeholder" style="margin-top: 2rem;">
                <div class="icon">💸</div>
                <h4 style="margin: 0;">Total Amount</h4>
                <p style="margin: 0.5rem 0 0; opacity: 0.8;">Will calculate automatically</p>
            </div>
            """, unsafe_allow_html=True)

        
        # Action buttons
        col_discard, col_clear = st.columns(2)
        with col_discard:
            if st.button("🗑️ Discard Last", key="discard_last_button"):
                if st.session_state.persistent_items:
                    st.session_state.persistent_items.pop()
        
        with col_clear:
            if st.button("🚮 Clear All", key="clear_all_button"):
                st.session_state.persistent_items.clear()
                
                if 'cap' in st.session_state:
                    st.session_state.cap.release()
                    del st.session_state.cap
                if 'current_source' in st.session_state:
                    del st.session_state.current_source
                st.session_state.uploader_key=str(time.time())
    
    # Main processing loop
    if 'model' in st.session_state and st.session_state.model and 'cap' in st.session_state:
        try:
            video_active = True
            while video_active and not st.session_state.stop_detection and st.session_state.cap.isOpened():
                t_start = time.perf_counter()
                
                ret, frame = st.session_state.cap.read()
                if not ret:
                    if source_type == "Upload Video":
                        st.warning("Video ended")
                    else:
                        st.error("Camera disconnected")
                    video_active = False
                    break
                
                # Resize frame
                frame = cv2.resize(frame, (resW, resH))
                
                # Process the frame for detection
                processed_frame = process_frame(frame, st.session_state.model, min_thresh)
                
                # Convert to RGB for Streamlit
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                if frame_placeholder is not None:
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Calculate billing information
                item_counts = defaultdict(int)
                total_amount = 0
                
                for item in st.session_state.persistent_items:
                    item_counts[item] += 1
                
                # Create billing table
                bill_data = []
                for item, count in item_counts.items():
                    if item in classes:
                        price = classes[item]
                        item_total = price * count
                        total_amount += item_total
                        bill_data.append({"Item": item, "Quantity": count, "Unit Price": f"Rs.{price}", "Total": f"Rs.{item_total}"})
                
                # Display bill
                bill_df = pd.DataFrame(bill_data)
                if not bill_df.empty:
                    bill_placeholder.dataframe(bill_df, use_container_width=True)
                else:
                    bill_placeholder.info("No items detected yet")
                
                # Display total
                total_placeholder.markdown(f"### Total Amount: Rs.{total_amount}")
                
                # Calculate FPS
                t_stop = time.perf_counter()
                frame_rate_calc = 1/(t_stop - t_start)
                
                if len(st.session_state.frame_rate_buffer) >= fps_avg_len:
                    st.session_state.frame_rate_buffer.pop(0)
                st.session_state.frame_rate_buffer.append(frame_rate_calc)
                st.session_state.avg_frame_rate = np.mean(st.session_state.frame_rate_buffer)
                
                # Allow Streamlit to update UI and handle events
                time.sleep(0.01)  # Adjust sleep time as needed
                
            # Loop ends here; no rerun needed
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            if 'cap' in st.session_state:
                st.session_state.cap.release()
                del st.session_state.cap
            if 'current_source' in st.session_state:
                del st.session_state.current_source
    if st.session_state.get("current_source") == "upload" or source_type == "Upload Video":
        if "persistent_items" in st.session_state and st.session_state.persistent_items:
            item_counts = defaultdict(int)
            total_amount = 0
            for item in st.session_state.persistent_items:
                item_counts[item] += 1

            bill_data = []
            for item, count in item_counts.items():
                if item in classes:
                    price = classes[item]
                    item_total = price * count
                    total_amount += item_total
                    bill_data.append({
                        "Item": item,
                        "Quantity": count,
                        "Unit Price": f"Rs.{price}",
                        "Total": f"Rs.{item_total}"
                    })
            
            pdf_data = generate_pdf_bill(bill_data, total_amount)
            
            # Updated download button with proper cleanup
            if st.download_button(
                label="Print Bill",
                data=pdf_data,
                file_name="bill.pdf",
                mime="application/pdf",
                key="final_download_button",
                on_click=lambda: [
                    st.session_state.persistent_items.clear(),
                    st.session_state.pop('cap', None),
                    st.session_state.pop('current_source', None),
                    st.session_state.__setattr__('uploader_key', str(time.time()))
                ]
            ):
                st.rerun()

if __name__ == "__main__":
    main()