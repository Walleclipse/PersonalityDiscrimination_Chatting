/**
 * Created by Jill on 16/11/19.
 * @author :  jill
 * @jill's blog : http://blog.csdn.net/jill6693484
 */

function chat(element,imgSrc,doctextContent){
    var $user = element;
    var $doctorHead = imgSrc;
    //获取输入的值
    var $textContent = $('.chat-info').html();
    //获取传入的医生输入的内容
    var $doctextContent =  doctextContent;
    //获取容器
    var $box = $('.bubbleDiv');
    var $boxHeght = $box.height();
    var $sectionHeght = $(".chat-box").height();
    var $elvHeght = Math.abs($boxHeght-$sectionHeght);
    //医生
    if($user === "leftBubble") {
        $box.append(createdoct(imgSrc,$doctextContent)).animate({scrollTop: $(document).height()}, 300);
    }
    //患者
    else if($user === "rightBubble") {
        $box.append(createuser($textContent)).animate({scrollTop: $(document).height()}, 300);      
        getClienttoServer($textContent)
        //test()
    }else {
        console.log("please say something～～");
    }

};


function finishChat() {
    $.ajax({  
        type:'get',      
        url:'http://54.245.160.212:5000/StopDialogue',  
          
        dataType:'json',  
        success:function(data) { 
            var $box = $('.bubbleDiv') 
            var response = data.Result
            if(response == 1){
                $box.append(createdoct('images/head_portrait.png',"see you~~")).animate({scrollTop: $(document).height()}, 300);
            }
         
     },   
    });

}


function getClienttoServer(formParam){ 
    //console.log(formParam)  
    $.ajax({  
        type:'post',      
        url:'http://54.245.160.212:5000/dialogue',  
        data:{'text':formParam},  
        cache:false,  
        dataType:'json',  
        success:function(data) {  
            var response = data.Result
            chat("leftBubble", "images/head_portrait.png",response)
         
     },   
    });

    $.ajax({  
        type:'get',      
        url:'http://54.245.160.212:5000/MBTIScore',     
        dataType:'json',  
        success:function(returneddata) { 
            var responses = returneddata.Result
            var ctx = document.getElementById("myChart");
            var myChart = new Chart(ctx, {
            type: 'bar',
    data: {
        labels: ['Introversion','Extroversion','Intuition','Sensing','Thinking','Feeling','Judging','Perceiving'],
        datasets: [{
            label: 'Your Personality Analysis',
            
            data: [responses[0]-1,responses[0],responses[1]-1,responses[1],responses[2]-1,responses[2],responses[3]-1,responses[3]],
            //data:responses
            backgroundColor: [
                'rgba(255, 99, 132, 0.2)',
                'rgba(255, 99, 132, 0.2)',
                'rgba(54, 162, 235, 0.2)',
                'rgba(54, 162, 235, 0.2)',
                'rgba(255, 206, 86, 0.2)',
                'rgba(255, 206, 86, 0.2)',
                'rgba(75, 192, 192, 0.2)',
                'rgba(75, 192, 192, 0.2)'
            ],
            borderColor: [
                'rgba(255,99,132,1)',
                'rgba(255,99,132,1)',
                'rgba(54, 162, 235, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(75, 192, 192, 1)'
            ],
            borderWidth: 1
        }]
    },
    options: {
        scales: {
            yAxes: [{
                ticks: {
                    beginAtZero:true
                }
            }]
        }
    }
});

            


            
     },   
    });


};
    






function createdoct(imgSrc, $doctextContent ) {
    var $textContent = $doctextContent;
    var $imgSrc = imgSrc;
    var block;
    if($textContent == ''|| $textContent == 'null'){
        alert('please say something～～');
        return;
    }
    block = '<div class="bubbleItem">' +
            '<div class="doctor-head">' +
            '<img src="'+ imgSrc +'" alt="doctor"/>' +
            '</div>' +
            '<span class="bubble leftBubble">' + $textContent + '<span class="topLevel"></span></span>' +
            '</div>';

    return block;
};

function createuser($textContent ) {
    var $textContent = $textContent;
    var block;
    if($textContent == ''|| $textContent == 'null'){
        alert('please say something～～');
        return;
    }
    block = '<div class="bubbleItem clearfix">' +
            '<span class="bubble rightBubble">' + $textContent + '<span class="topLevel"></span></span>' +
            '</div>';

    return block;
};

