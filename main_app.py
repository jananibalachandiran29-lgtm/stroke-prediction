import pickle
import pandas as pd


def main():
    print("=" * 55)
    print("          AI-POWERED STROKE RISK PREDICTION")
    print("=" * 55)

    # Load trained model
    try:
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        print("\nError: model.pkl not found.")
        print("Please run train.py first.")
        return

    print("\nEnter patient information:\n")

    try:
        gender = input("Gender (Male/Female/Other): ")
        age = float(input("Age: "))
        hypertension = int(input("Hypertension (0 = No, 1 = Yes): "))
        heart_disease = int(input("Heart Disease (0 = No, 1 = Yes): "))
        ever_married = input("Ever Married (Yes/No): ")
        work_type = input("Work Type: ")
        residence_type = input("Residence Type (Urban/Rural): ")
        avg_glucose_level = float(input("Average Glucose Level: "))
        bmi = float(input("BMI: "))
        smoking_status = input(
            "Smoking Status (formerly smoked/never smoked/smokes/Unknown): "
        )

    except ValueError:
        print("\nInvalid input. Please enter the correct values.")
        return

    # Create patient data
    patient = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }])

    # Make prediction
    try:
        prediction = model.predict(patient)[0]
        probability = model.predict_proba(patient)[0][1]

        print("\n" + "=" * 55)
        print("                    RESULT")
        print("=" * 55)

        print(f"Predicted stroke probability: {probability * 100:.2f}%")

        if prediction == 1:
            print("Prediction: Stroke-risk class")
        else:
            print("Prediction: No-stroke class")

        # Heuristic safety rule
        if age >= 65 and hypertension == 1 and heart_disease == 1:
            print("\nHeuristic Alert: High-risk profile detected.")
            print("Multiple risk factors require professional medical attention.")

        print("=" * 55)
        print("\nDisclaimer:")
        print("This project is for academic and educational purposes only.")
        print("It is not a medical diagnostic system.")

    except Exception as error:
        print("\nPrediction error:", error)


if __name__ == "__main__":
    main()
