from flask import Flask,request,jsonify,send_file,render_template
import re,nltk
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import base64
import traceback

nltk.download('stopwords')
STOPWORDS=set(stopwords.words("english"))
app=Flask(__name__)

@app.route("/test",methods=["GET"])
def test():
    return "Test request recevied sucessfully"

@app.route("/",methods=["GET","POST"])
def home():
    return render_template("landing.html")

@app.route("/predict",methods=["POST"])
def predict():
    try:
        predictor=pickle.load(open(r"Models/model_xgb.pkl","rb"))
        scaler=pickle.load(open(r"Models/scaler.pkl","rb"))
        cv=pickle.load(open(r"Models/countVectorizer.pkl","rb"))
   
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            data = pd.read_csv(file)
            predictions, graph = bulk_prediction(predictor, scaler, cv, data)

            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv"
            )
            response.headers['X-Graph-Exists'] = "true"
            response.headers['X-Graph-Data'] = base64.b64encode(graph.getbuffer()).decode("ascii")
            return response

        elif request.is_json and "text" in request.json:
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
            return jsonify({"prediction": predicted_sentiment})

        return jsonify({"error": "Invalid input"}), 400

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 500

def single_prediction(predictor,scaler,cv,text_input):
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]"," ",text_input)
    review= review.lower().split()
    review=[stemmer.stem(word)for word in review if not word in STOPWORDS]
    review=" ".join(review)
    processed = " ".join(review)

    X_prediction = cv.transform([processed]).toarray()
    X_prediction_scl =scaler.transform(X_prediction)
    y_predictions =predictor.predict_proba(X_prediction_scl)
    y_predictions= y_predictions.argmax(axis=1)[0]
    return "postive" if y_predictions== 1 else "Negative"
    
def bulk_prediction(predictor,scaler,cv,data):
    corpus=[]
    stemmer=PorterStemmer()
    for i in range(0,data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ",data.iloc[i]["sentence"])
        review= review.lower().split()
        review=[stemmer.stem(word)for word in review if not word in STOPWORDS]
        corpus.append(" ".join(review))

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl =scaler.transform(X_prediction)
    y_predictions =predictor.predict_proba(X_prediction_scl).argmax(axis=1)
    y_predictions=list(map(sentiment_mapping,y_predictions))

    data["predicted sentiment"]= y_predictions
    predictions_csv = BytesIO()
    data.to_csv("Predictions.csv",index=False)
    predictions_csv.seek(0)
    graph = get_distribution_graph(data)
    return predictions_csv,graph

def get_distribution_graph(data):
    fig=plt.figure(figsize=(5,5))
    colors=("pink","red")
    wp={"linewidth":1,"edgecolor":"black"}
    tags=data["predicted sentiment"].value_counts()
    explode=(0.01,0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow="True",
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
    )
    plt.title("sentimentat Distributation")
    graph=BytesIO()
    plt.savefig(graph,format="png")
    plt.close()
    graph.seek(0)
    return graph
def sentiment_mapping(x):
    if x==1:
        return "positive"
    else:
        return "Negative"

if __name__ == "__main__":
    app.run(port=5000,debug=True)