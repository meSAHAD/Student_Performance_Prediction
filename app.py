
import gradio as gr
import pandas as pd
import joblib

# Load model
model = joblib.load("student_model.pkl")

# Prediction function
def predict_result(
    age,
    tuition_fee,
    time_friends,
    ssc_result
):

    input_df = pd.DataFrame([[
        age,
        tuition_fee,
        time_friends,
        ssc_result
    ]],

    columns=[
        'age',
        'tuition_fee',
        'time_friends',
        'ssc_result'
    ])

    prediction = model.predict(input_df)[0]

    if prediction >= 4.5:
        status = "Excellent Performance"

    elif prediction >= 3.5:
        status = "Good Performance"

    elif prediction >= 2.5:
        status = "Average Performance"

    else:
        status = "Needs Improvement"

    return f"""
Predicted HSC GPA: {prediction:.2f}

Performance Status: {status}
"""

# Interface
app = gr.Interface(
    fn=predict_result,

    inputs=[

        gr.Number(label="Student Age"),

        gr.Number(label="Tuition Fee"),

        gr.Number(label="Time With Friends"),

        gr.Number(label="SSC GPA")

    ],

    outputs="text",

    title="Student Performance Prediction System"
)

app.launch()
