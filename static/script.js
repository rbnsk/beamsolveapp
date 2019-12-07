inputs = {
    BeamLen: [],
    PL: [],
    UDL: [],
    Mom: [],
    Pin: [],
    Roller: [],
    Fixed: []
}

//Canvas

var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d')

const renderCanvas = function(){

    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.width = canvas.scrollWidth
    canvas.height = 200

    //Beam
    BeamLength = 800
    BeamHeight = 5

    beamY = canvas.height / 2
    beamX = (canvas.width - BeamLength) / 2
    BeamSegment = BeamLength / inputs.BeamLen
    ctx.fillStyle = "#6D6D6D"
    ctx.fillRect(beamX,beamY - 10,BeamLength,BeamHeight)

    //Length

    ctx.fillRect(beamX,beamY + 50,BeamLength,2)
    ctx.fillRect(beamX,beamY + 45,2,13)
    ctx.fillRect(beamX + BeamLength,beamY + 45,2,13)
    ctx.fillStyle = 'black'
    ctx.font = "15px Arial";
    ctx.fillText(inputs.BeamLen + 'm', beamX + (BeamLength/2 - 5), beamY + 70);

    //Pin Support
    inputs.Pin.forEach(function(value) {
        
        ctx.beginPath();
        ctx.fillStyle = "#ff7d7d"
        startingX = beamX + (BeamSegment * value)
        startingY = beamY + BeamHeight - 10
        ctx.moveTo(startingX, startingY); // pick up "pen," reposition at 500 (horiz), 0 (vert)
        ctx.lineTo(startingX - 20, startingY + 30); // draw straight down by 200px (200 + 200)
        ctx.lineTo(startingX + 20, startingY + 30)
        ctx.closePath(); // connect end to start
        ctx.fill();
    });

    //Roller Support
    inputs.Roller.forEach(function(value) {
        
        ctx.beginPath();
        ctx.fillStyle = "#ff7d7d"
        startingX = beamX + (BeamSegment * value)
        startingY = beamY + BeamHeight + 5
        ctx.arc(startingX, startingY, 15, 0, Math.PI * 2, true);
        ctx.fill();
    });

    //Fixed Support
    inputs.Fixed.forEach(function(value) {
        
        if (value == "Right"){
            ctx.fillStyle = "#ff7d7d"
            ctx.fillRect(beamX + BeamLength,beamY+ 30,10,-70)
        }

        else if (value == "Left"){
            ctx.fillStyle = "#ff7d7d"
            ctx.fillRect(beamX,beamY+ 30,-10,-70)
        }

    });

    //ConcLoad
    inputs.PL.forEach(function(value) {
        startingX = beamX + (BeamSegment * value.location)

        ctx.font = "15px Arial";
        ctx.fillStyle = 'black'
        if (value.load > 0) {
            ctx.fillText(value.load + 'N', startingX - 10, 15 );
            ctx.beginPath();
            drawArrow(startingX, 22, startingX, beamY + BeamHeight - 17);
            ctx.stroke();
        }

        else if (value.load < 0){
            ctx.fillText(value.load + 'N', startingX - 10, 15 );
            ctx.beginPath();
            drawArrow(startingX, beamY + BeamHeight - 15, startingX, 22);
            ctx.stroke();
        }
        
    });

    //Moments
    inputs.Mom.forEach(function(value) {
        ctx.beginPath();
        ctx.arc(5, 5, 5, 0, Math.PI, false);
        ctx.closePath();
    });

}

function drawArrow(fromx, fromy, tox, toy){
    //variables to be used when creating the arrow
    var c = document.getElementById("canvas");
    var ctx = c.getContext("2d");
    var headlen = 10;

    var angle = Math.atan2(toy-fromy,tox-fromx);

    //starting path of the arrow from the start square to the end square and drawing the stroke
    ctx.beginPath();
    ctx.moveTo(fromx, fromy);
    ctx.lineTo(tox, toy);
    ctx.strokeStyle = "#00C7A9";
    ctx.lineWidth = 4;
    ctx.stroke();

    //starting a new path from the head of the arrow to one of the sides of the point
    ctx.beginPath();
    ctx.moveTo(tox, toy);
    ctx.lineTo(tox-headlen*Math.cos(angle-Math.PI/7),toy-headlen*Math.sin(angle-Math.PI/7));

    //path from the side point of the arrow, to the other side point
    ctx.lineTo(tox-headlen*Math.cos(angle+Math.PI/7),toy-headlen*Math.sin(angle+Math.PI/7));

    //path from the side point back to the tip of the arrow, and then again to the opposite side point
    ctx.lineTo(tox, toy);
    ctx.lineTo(tox-headlen*Math.cos(angle-Math.PI/7),toy-headlen*Math.sin(angle-Math.PI/7));

    //draws the paths created above
    ctx.strokeStyle = "#00C7A9";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = "#00C7A9";
    ctx.fill();
}

//Display Model
const modalOpen = function(type){
    var modal = document.getElementById('modal' + type);
    modal.style.display = 'block'
}

//Close Modal
const modalCancel = function(type){
    var modal = document.getElementById('modal' + type);
    modal.style.display = 'none'
}

//RenderPills
const renderPills = function(type){
    var container = document.getElementById("container" + type);
    container.innerHTML = "";
    inputs[type].forEach(function(value,index){
        var pill = document.createElement("span");
        pill.classList.add("tag","is-warning","is-rounded","is-small");
        
        if (type === 'PL'){
            var location = value.location
            var load = value.load
            var deleteButton = "<button class='delete' 'is-small' onclick= 'deletePill(`PL`," + index + ")'></button>"
            pill.innerHTML = "PL " + (index +1)+ ": " + load + "kN @ " + location + "m" + deleteButton
        }
        
        else if (type === 'UDL'){
            var load = value.load
            var start = value.start
            var end = value.end
            var deleteButton = "<button class='delete' 'is-small' onclick= 'deletePill(`UDL`," + index + ")'></button>"
            pill.innerHTML = "UDL " + (index +1)+ ": " + load + "kN/m @ " + start + "m to " + end + "m" + deleteButton
        }

        else if (type === 'Mom'){
            var location = value.location
            var load = value.load
            var deleteButton = "<button class='delete' 'is-small' onclick= 'deletePill(`Mom`," + index + ")'></button>"
            pill.innerHTML = "Moment " + (index +1)+ ": " + load + "kNm @ " + location + "m" + deleteButton
        }

        else if (type === 'Pin'){
            var location = value
            var deleteButton = "<button class='delete' 'is-small' onclick= 'deletePill(`Pin`," + index + ")'></button>"
            pill.innerHTML = "Pin " + (index +1)+ " @ " + location + "m" + deleteButton 
        }

        else if (type === 'Roller'){
            var location = value
            var deleteButton = "<button class='delete' 'is-small' onclick= 'deletePill(`Roller`," + index + ")'></button>"
            pill.innerHTML = "Roller " + (index +1)+ " @ " + location + "m" + deleteButton  
        }
        
        else if (type === 'Fixed'){

            var position = ""
            if(value == "Right"){
                position = "Right"
            }
            else if (value == "Left"){
                position = "Left"
            }
            var deleteButton = "<button class='delete' 'is-small' onclick= 'deletePill(`Fixed`," + index + ")'></button>"
            pill.innerHTML = "Fixed @ " + position + deleteButton
            
        }
        container.appendChild(pill)
    })
}

const deletePill = function(type,index){
    
    if (type === 'PL'){
        inputs.PL.splice(index, 1)
    }
    else if (type === 'UDL'){
        inputs.UDL.splice(index, 1)
    }

    else if (type === 'Mom'){
        inputs.Mom.splice(index, 1)
    }

    else if (type === 'Pin'){
        inputs.Pin.splice(index, 1)
    }

    else if (type === 'Roller'){
        inputs.Roller.splice(index, 1)     
    }
    
    else if (type === 'Fixed'){
        inputs.Fixed.splice(index, 1)
    }

    renderPills(type)
    renderCanvas()
}

// Beam Length
document.querySelector('#addBeamLen').addEventListener('click',function() {
    inputs.BeamLen = document.getElementById("inputBeamLen").value;
    
    var caption = document.getElementById("caption");
    
    if (caption !== null){
        caption.parentNode.removeChild(caption);
    }
    
    canvas.style.display= "block"
    renderCanvas()
})


//Forces

//Point Load 
document.querySelector('#addPL').addEventListener('click',function() {
    modalOpen('PL')
})

document.querySelector('#addPLModal').addEventListener('click',function(e) {
    e.preventDefault()
    var load = document.getElementById("PLmag").value
    var location = document.getElementById("PLlocation").value
    inputs.PL.push({
        load: load,
        location: location
    })
   renderPills('PL')
   renderCanvas()
   modalCancel('PL')
})


//UDL

document.querySelector('#addUDL').addEventListener('click',function() {
    modalOpen('UDL')
})

document.querySelector('#addUDLModal').addEventListener('click',function(e) {
    e.preventDefault()
    var start = document.getElementById("UDLstart").value
    var end = document.getElementById("UDLend").value
    var load = document.getElementById("UDLmag").value
    inputs.UDL.push({
        start: start,
        end: end,
        load: load,
    })
    renderPills('UDL')
    renderCanvas()
    modalCancel('UDL')
})


//Moments

document.querySelector('#addMom').addEventListener('click',function() {
    modalOpen('Mom')
})

document.querySelector('#addMomModal').addEventListener('click',function(e) {
    e.preventDefault()
    var load = document.getElementById("Mommag").value
    var location = document.getElementById("Momlocation").value
    inputs.Mom.push({
        load: load,
        location: location
    })
   renderPills('Mom')
   renderCanvas()
   modalCancel('Mom')
})

//Support

//Pin

document.querySelector('#addPin').addEventListener('click',function() {
    modalOpen('Pin')
})

document.querySelector('#addPinModal').addEventListener('click',function(e) {
    e.preventDefault()
    inputs.Pin.push(document.getElementById("pinLocation").value)
    renderPills('Pin')
    renderCanvas()
    modalCancel('Pin')
})

//Roller

document.querySelector('#addRoller').addEventListener('click',function() {
    modalOpen('Roller')
})

document.querySelector('#addRollerModal').addEventListener('click',function(e) {
    e.preventDefault()
    inputs.Roller.push(document.getElementById("rollerLocation").value)
    renderPills('Roller')
    renderCanvas()
    modalCancel('Roller')
})

//Fixed

document.querySelector('#addFixed').addEventListener('click',function() {
    modalOpen('Fixed')
})

document.querySelector('#addFixedModal').addEventListener('click',function(e) {
    e.preventDefault()
    inputs.Fixed = []
    if (document.getElementById("FixLeft").checked){
        inputs.Fixed.push("Left")
    }

    if (document.getElementById("FixRight").checked ){
        inputs.Fixed.push("Right")
    }
    if (document.getElementById("FixRight").checked == false && document.getElementById("FixLeft").checked == false) {
        inputs.Fixed = []
    }

    renderPills('Fixed')
    renderCanvas()
    modalCancel('Fixed')
    console.log(inputs)
})

//Generate

document.querySelector('#generate').addEventListener('click',function() {

    var json_inputs = JSON.stringify(inputs);
    document.getElementById("graphs").classList.remove("is-hidden")

    var bmGraph = document.getElementById("bmGraph");;
    bmGraph.innerHTML = "";

    var shearGraph = document.getElementById("shearGraph");;
    shearGraph.innerHTML = "";

    $.ajax({
        type: "POST",
        contentType: "application/json;charset=utf-8",
        url: "/",
        traditional: "true",
        data: json_inputs,
        dataType: "json",
        success: function(data) {
            
            var image = new Image();
            image.src = "data:image/png;base64," + data[0]
            bmGraph.appendChild(document.body.appendChild(image))
            bmGraph.style.textAlign = "center"
            bmGraph.appendChild(document.createTextNode("Max Bending Moment is " + data[2].toFixed(2) + "Nm"))

            var image2 = new Image();
            image2.src = "data:image/png;base64," + data[1]
            shearGraph.appendChild(document.body.appendChild(image2))
            shearGraph.style.textAlign = "center"
            shearGraph.appendChild(document.createTextNode("Max Shear Force is " + data[3].toFixed(2) + "N"))
    
            console.log(data[2])
            
            console.log(data[3])
            }    
    })
})
