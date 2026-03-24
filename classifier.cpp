#include <iostream>
using namespace std;
#include "csvstream.hpp"
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <cmath>

class Classifier {
public:
    void train(csvstream &csv);
    string predict(const set<string> &words) const;
    // helpers for printing training info
    int getNumPosts() const;
    int getVocabSize() const;
    void setNumPosts()
    const map<string,int>& getPostsPerLabel() const;
    const map<pair<string,string>,int>& getLabelWordCounts() const;
    bool trainingMode = false;
    double getScore(const set<string> &words, const string &label) const;
private:
    int numPosts = 0;
    int numUniqueWords;

    map<string, string> posts;  // each post and its content
    map<string, int> wordPostCount;         // word -> # posts containing it
    map<string, int> postsPerLabel;         // label -> # posts with that label
    map<pair<string,string>, int> labelWordCount; // {label, word} -> # posts
    set<string> vocab; 
};

int main(int argc, char *argv[]) {

    Classifier cl;

    // check how many arguments there are
    // If theres 3 or less arguments its just training mode
    cl.trainingMode = (argc == 2);

    // read in the data
    // Open the file
    csvstream csvin(argv[1]);
   try(csvin) {
      cout << "Error opening file: " << argv[1] << endl;
      return 1;
    }
    cl.train(csvin);

    // read in the test file and run test
    if(!trainingMode){
      cout << "test data:" << endl;
      int correct = 0; 
      int total = 0;

      csvstream testIn(argv[2]);
      map<string, string> row;
      while(testIn >> row){
        string correctLabel = row["tag"];
        set<string> words = unique_words(row[content]);
        string predicted = cl.predict(words);
        double score = cl.getScore(words, predicted);

        cout << "  correct = " << correctLabel << ", predicted = " << predicted
             << ", log-probability score = " << score << endl;
        cout << "  content = " << row["content"] << endl << endl; 

        if (correctLabel == predicted) correct++;
        total++;
      }
      cout << "performance: " << correct << " / " << total
        << " posts predicted correctly" << endl;
    }
    
  
}

// train model
void Classifier::train(csvstream &csv) {
  if(trainingMode) cout << "training data: " << endl;

  // Each row comes back as a map<string, string>
  map<string, string> row;
  while (csv >> row) {
    string tag = row["tag"];
    // Seperated words per post 
    
    set<string> words = unique_words(row["content"]);

    if(trainingMode) cout << "label = " << tag << ", content = " << row["content"] << endl;
    
    // add a new post
    numPosts++;
    // increment the current label 
    postsPerLabel[tag]++;
    
    // loop through each word
    for (const string &word : words) {
      // insert a new vocab word (won't repeat)
      vocab.insert(word);
      // increment the num times that word is in the post
      wordPostCount[word]++;
      // increment num times that word is in that tag and post
      labelWordCount[{tag, word}]++;
    }
  }
  if (trainingMode) { 
    cout << "trained on " << numPosts << " examples" << endl;
    cout << "vocabulary size = " << vocab.size() << endl << endl;
  }

  if (trainingMode) {
    cout << "classes:" << endl;
    for (auto &[label, count] : postsPerLabel) {
      double logPrior = log((count) / numPosts);
      cout << "  " << label << ", " << count << " examples, log-prior = " << logPrior << endl;
    }
    cout << "classifier parameters:" << endl;
    for (auto &[labelWord, count] : labelWordCount) {
      double logLikelihood = log((count) / postsPerLabel[labelWord.first]);
      cout << "  " << labelWord.first << ":" << labelWord.second
        << ", count = " << count << ", log-likelihood = " << logLikelihood << endl;
    }
    cout << endl;
  }
}

string Classifier::predict(const set<string> &words){

  // Predict (test)
  // for each label which has the highest log-prob score for the post
  // in case of tie first alphabetically
  // if w was never seen in a post with abel C in training
    // if does occur in overall training
    // If does not occur anywhere in training
  string bestLabel = "";
  double bestScore = 0;

  // loop over every label
  for (auto &[label, labelCount] : postsPerLabel) {
    double score = getScore(words, label);
    
    if(score > bestScore) {
      bestScore = score;
      bestLabel = label;
    }
  }
  return bestLabel;
}


// EFFECTS: Return a set of unique whitespace delimited words
set<string> unique_words(const string &str) {
  istringstream source(str);
  set<string> words;
  string word;
  while (source >> word) {
    words.insert(word);
  }
  return words;
}

double Classifier::getScore(const set<string> &words, const string &label) const {
    int labelCount = postsPerLabel.at(label);
    
    // start with log-prior
    double score = log(static_cast<double>(labelCount) / numPosts);
    
    // add log-likelihood for each word
    for (const string &word : words) {
        if (labelWordCount.count({label, word})) {
            // word seen with this label
            score += log(static_cast<double>(labelWordCount.at({label, word})) / labelCount);
        } else if (wordPostCount.count(word)) {
            // word in training but not with this label
            score += log(static_cast<double>(wordPostCount.at(word)) / numPosts);
        } else {
            // word never seen in training
            score += log(1.0 / numPosts);
        }
    }
    return score;
}