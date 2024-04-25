import gradio as gr  # استيراد مكتبة Gradio لإنشاء الواجهة
import tensorflow as tf  # استيراد TensorFlow لاستخدام النموذج
import numpy as np  # استيراد NumPy لعمليات الصفائف

# تحميل النموذج المدرب مسبقًا والمحفوظ كملف 'model.h5'
model = tf.keras.models.load_model('model.h5')

# الدالة للتعرف على الرقم في الصورة الواردة
def recognize_digit(image):
    # التحقق مما إذا كانت الصورة الواردة ليست فارغة
    if image is not None:
        # تحويل الصورة إلى صفيف NumPy وتغيير شكلها إلى (1, 28, 28, 1) وتطبيع القيم
        image = np.array(image).reshape((1, 28, 28, 1)).astype('float32') / 255
        # استخدام النموذج المحمل للتنبؤ بالرقم في الصورة
        prediction = model.predict(image)
        # إرجاع احتماليات التنبؤ لكل رقم كقاموس
        return {str(i): float(prediction[0][i]) for i in range(10)}
    else:
        # إرجاع سلسلة فارغة إذا كانت الصورة فارغة
        return ''

# إنشاء واجهة Gradio لوظيفة التعرف
iface = gr.Interface(
    fn=recognize_digit,  # استخدام الدالة recognize_digit كدالة معالجة
    inputs=gr.inputs.Image(shape=(28, 28), image_mode='L', invert_colors=True, source='canvas'),  # تعريف خصائص الصورة الواردة
    outputs=gr.outputs.Label(num_top_classes=3),  # تعريف خصائص العلامة الناتجة
    live=True  # تمكين التحديثات `الحية في الواجهة
)

# إطلاق واجهة Gradio
iface.launch()
