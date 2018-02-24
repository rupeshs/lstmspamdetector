"use strict";
const SEQUENCE_LENGTH = 654;
var wordsCount = 0;
var dict = {};
var model;

function TextToIndexVector(text, seqlen) {

	var wordVec = new Array();
	text=text.toLowerCase();
	var wordlst = text.split(/\s+/);
	wordsCount = wordlst.length;

	wordlst.forEach(function (word) {

		if (wordVec.length != seqlen) {
			if (word in dict) {
				wordVec.push(dict[word])
				
			} else {
				wordVec.push(0)
			}
		}

	});
	var wordVecPadded = wordVec.concat(Array(seqlen).fill(0)).slice(0, seqlen);
	//console.log(wordVecPadded);
	return wordVecPadded;
}
function LoadModel() {
NProgress.start();
$("#wordCnt").html("Loading model,please wait...");
	$.getJSON("wordindex.json", function (json) {
		
		dict = json;
		});
	model = new KerasJS.Model({
		filepath: 'spam_lstm_model.bin',
		gpu: false
	});

	model
		.ready()
		.then(() => {
			console.log("Model ready");
			NProgress.done();
			$("#messageText").prop('disabled', false);
			$("#wordCnt").html("Words : " + wordsCount);
		})


}
function checkInput()
{
	var x = document.getElementById("messageText").value;
	if (x=="")
	{   wordsCount=0;
        $("#detectBtn").prop('disabled', true);
        $("#messageType").html("<div class=\"alert alert-warning\">Waiting for input...</div>");
        $("#wordCnt").html("Words : " + wordsCount);
		return;
	}
	else{
		$("#detectBtn").prop('disabled', false);
	}
}
function predictSpam() {
	var x = document.getElementById("messageText").value;
	if (x=="")
		return;

	var result = TextToIndexVector(x, SEQUENCE_LENGTH);
	var seqIn = new Float32Array(result);

	model.predict({
		input: seqIn
	}).then(outputData => {
		/*
		0   1
		Ham Spam
		*/
		if (outputData.output[0] > outputData.output[1]) {
			
			$("#messageType").html("<div class=\"alert alert-success\"> <h5><i  class=\"fa fa-check\"></i> Not spam.</h5></div>");
		} else {
		
			$("#messageType").html("<div class=\"alert alert-danger\"><h5> <i  class=\"fa fa-ban\"></i> Oops...Spam!</h5></div>");
		}
		$("#wordCnt").html("Words : " + wordsCount);
	});

}