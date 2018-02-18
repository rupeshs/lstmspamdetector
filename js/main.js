"use strict";

const SEQUENCE_LENGTH = 30;
var wordsCount = 0;
var dict = {};
var model;

function TextToIndexVector(text, seqlen) {

	var wordVec = new Array();
	var wordlst = text.split(/\s+/);
	wordsCount = wordlst.length;

	wordlst.forEach(function (word) {

		if (wordVec.length != seqlen) {
			if (word in dict) {
				wordVec.push(dict[word])
				console.log(dict[word] + " " + word);
			} else {
				wordVec.push(0)
			}
		}

	});
	var wordVecPadded = wordVec.concat(Array(seqlen).fill(0)).slice(0, seqlen);
	console.log(wordVecPadded);
	return wordVecPadded;
}


function LoadModel() {
NProgress.start();
$("#wordCnt").html("Loading model,please wait...");
	$.getJSON("wordindex.json", function (json) {
		console.log(json); 
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


function predictSpam() {

	var x = document.getElementById("messageText").value;
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
		
			$("#messageType").html("<div class=\"alert alert-danger\"><h5> <i  class=\"fa fa-ban\"></i> Oops...it is a spam!</h5></div>");
		}
		$("#wordCnt").html("Words : " + wordsCount);
	});

}